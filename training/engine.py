"""
Contains functions for training and testing a PyTorch model.
"""
import shutil
import torch
import time
import utils
import numpy as np
import copy
from tqdm import trange
import os

class Trainer:
    def __init__(self, 
                 model: torch.nn.Module,
                 dataloaders: torch.utils.data.DataLoader,
                 epochs: int, 
                 metric: torch.nn.Module, 
                 criterion: torch.nn.Module, 
                 optimizer: torch.optim.Optimizer,
                 scheduler: torch.optim.lr_scheduler,
                 save_dir: str,
                 device: torch.device,
                 save_checkpoint = False):

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.model = model
        self.checkpoint = save_checkpoint
        self.train_dataloader = dataloaders['train']
        self.val_dataloader = dataloaders['val']
        self.epoch = 0
        self.epochs = epochs
        self.metric = metric
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_dir = save_dir
        self.device = device
        self.weights = []
        self.IoUs = []

        # Create empty results dictionary
        self.results = {"train_loss": [],
                        "train_iou": [],
                        "val_loss": [],
                        "val_iou": []
                        }
        
    def train_model(self):
        """
        Train the Model.
        """
        start_time = time.time()

        progressbar = trange(self.epochs, desc="Progress")
        for _ in progressbar:
            # Epoch counter
            self.epoch += 1

            # Training block
            self.train_epoch()

            # Validation block
            self.val_epoch()
            print(f'\nEpoch {self.epoch}: Train loss: {self.results["train_loss"][-1]} Train iou: {self.results["train_iou"][-1]} Val loss: {self.results["val_loss"][-1]} Val iou: {self.results["val_iou"][-1]}')
            self.weights.append(copy.deepcopy(self.model.state_dict()))
            self.IoUs.append(self.results["val_iou"][-1])
            # Save checkpoints every 5 epoch
            if (self.epoch % 8 == 0) and self.checkpoint == True:
                utils.save_checkpoint(self.model, self.optimizer, self.scheduler, os.path.join(self.save_dir, "checkpoint"), f"checkpoint_{self.epoch}")
        utils.save_checkpoint(self.model, self.optimizer, self.scheduler, os.path.join(self.save_dir, "checkpoint"),
                              f"final_checkpoint_{self.epoch}")
        time_elapsed = time.time() - start_time
        print('\n')
        print('-' * 20)
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

        # plot training curve
        utils.plot_curve(results=self.results, epochs=self.epochs)

        return self.results

    def train_epoch(self):
        """
        Training Mode
        """
        self.model.train() # training mode
        running_ious, running_losses = [], []

        for x, y in self.train_dataloader:
            # Send to device (GPU or CPU)
            inputs = x.to(self.device)
            targets = y.to(self.device)

            # Zero the parameter gradients
            self.optimizer.zero_grad()

            # Forward - track history if only in train
            outputs = self.model(inputs)
            # Calculate the loss
            loss = self.criterion(outputs, targets)
            loss_value = loss.item()
            running_losses.append(loss_value)

            # Calculate the iou
            iou_value = self.metric(outputs, targets)
            running_ious.append(iou_value.item())

            # Backward pass
            loss.backward()
            # Update the parameters
            self.optimizer.step()

        self.scheduler.step()
        self.results["train_loss"].append(np.mean(running_losses))
        self.results["train_iou"].append(np.mean(running_ious))

    def val_epoch(self):
        """
        Validation Mode
        """
        self.model.eval() # Validation mode
        running_ious, running_losses = [], []

        for x, y in self.val_dataloader:
            # Send to device (GPU or CPU)
            inputs = x.to(self.device)
            targets = y.to(self.device)

            with torch.no_grad():
                outputs = self.model(inputs)
                # Calculate the loss
                loss = self.criterion(outputs, targets)
                loss_value = loss.item()
                running_losses.append(loss_value)

                # Calculate the iou
                iou_value = self.metric(outputs, targets)
                running_ious.append(iou_value.item())

        self.results["val_loss"].append(np.mean(running_losses))
        self.results["val_iou"].append(np.mean(running_ious))


    def save_best_model(self, name):
        index = np.argmax(self.IoUs)
        self.model.load_state_dict(self.weights[index])
        utils.save_model(self.model, self.save_dir, name)
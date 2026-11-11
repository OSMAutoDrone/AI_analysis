import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import keyboard
import shutil
import time
base_path = r"./data/"
viz_path = os.path.join(base_path, 'base/viz')
discarded_path = os.path.join(base_path, 'base/discarted')
img_path = os.path.join(base_path, 'base/img')
mask_path = os.path.join(base_path, 'base/mask')

if __name__ == '__main__':
    plt.rcParams['figure.figsize'] = [16, 8]
    if not os.path.exists(discarded_path):
        os.makedirs(discarded_path)
    for root, dirs, files in os.walk(viz_path):
        for file in files:
            viz = mpimg.imread(os.path.join(viz_path, file))
            plt.imshow(viz)
            plt.pause(0.1)
            key_pressed = False
            while not key_pressed:
                event = keyboard.read_event(suppress=True)
                if event.event_type != keyboard.KEY_DOWN:
                    continue
                if event.name == "gauche":
                    key_pressed = True
                    os.remove(os.path.join(img_path, file.replace('.png', '.jpg')))
                    os.remove(os.path.join(mask_path, file))
                    shutil.move(os.path.join(viz_path, file), os.path.join(discarded_path, file))
                if event.name == "droite":
                    key_pressed = True
                time.sleep(0.2)

            plt.close()


import cv2
import random
import numpy as np
import pandas as pd
import skimage as sk
from skimage import util
from scipy import ndarray
from skimage import transform
import matplotlib.pyplot as plt
import matplotlib.image as imgsave


def random_rotation(image_array):
    
    random_degree = random.uniform(-45, 45)
    return sk.transform.rotate(image_array, random_degree)

def random_noise(image_array):
    
    return sk.util.random_noise(image_array)

def horizontal_flip(image_array):
    
    return image_array[:, ::-1]

def vertical_flip(image_array):
    
    return image_array[::-1, :]

def flip(image_array):

    choice = random.choice([horizontal_flip, vertical_flip])
    return(choice(image_array))

def change_brightness(image_array):
    value = random.randint(-30, 30)
    value = np.uint8(value)
    hsv = cv2.cvtColor(image_array, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    if value >= 0:
        lim = 255 - value
        v[v > lim] = 255
        v[v <= lim] += value
    if value < 0:
        v[v > value] -= value
        v[v <= value] += 0

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img


available_transformations = {
    'rotate': random_rotation,
    'noise': random_noise,
    'flip': flip,
    'change_brightness': change_brightness
}

def augment_images(image_paths, labels):
    all_paths = []
    all_labels = []

    progress = 0
    total = len(labels)
    i = 0
    for image_path, label in zip(image_paths, labels):
        print('Progress', round(progress / total * 100, 3))
        progress += 1
        
        image_to_transform =  plt.imread(image_path)
        
        num_transformations_to_apply = random.randint(1, len(available_transformations))
        num_of_transformations_occured = 0
        
        imgsave.imsave('./augmented_image/' + str(i) + '.jpg', image_to_transform)
        all_paths.append('./augmented_image/' + str(i) + '.jpg')
        all_labels.append(label)

        while (num_of_transformations_occured <= num_transformations_to_apply):
            
            key = random.choice(list(available_transformations))
            transformed_image = available_transformations[key](image_to_transform)

            imgsave.imsave('./augmented_image/' + str(i) + '_' + str(num_of_transformations_occured) +'.jpg', transformed_image)

            all_paths.append('./augmented_image/' + str(i) + '_' + str(num_of_transformations_occured) +'.jpg')
            all_labels.append(label)

            num_of_transformations_occured += 1
        
        i += 1

    to_write = pd.DataFrame()
    to_write['image_id'] = all_paths
    to_write['dx'] = all_labels
    to_write.to_csv('./augmented_image/meta_augmented.csv')

    return(all_paths, all_labels)


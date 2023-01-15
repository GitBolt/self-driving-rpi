import os
import cv2
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.utils import shuffle
from imgaug import augmenters as iaa
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D, Flatten, Dense


# Init
def get_name(file_path: str) -> str:
    image_path_location = file_path.split('/')[-2:]
    image_path = os.path.join(image_path_location[0], image_path_location[1])
    return image_path


def import_data_info(path: str) -> pd.DataFrame:
    columns: list[str] = ['Center', 'Steering']
    folder_count: int = len(os.listdir(path)) // 2
    data = pd.DataFrame()
    for x in range(17, 22):
        new_data = pd.read_csv(
            os.path.join(path, f'log_{x}.csv'),
            names=columns
        )

        print(f'{x} : {new_data.shape[0]} ', end='')

        new_data['Center'] = new_data['Center'].apply(get_name)
        data = data.append(new_data, True)

    print('\nTotal images imported: ', data.shape[0])
    return data


# Visualize and balance
def balance_data(data: pd.DataFrame, display: bool = True):
    bin_count = 31
    samples_per_bin = 300
    hist, bins = np.histogram(data['Steering'], bin_count)
    if display:
        center = (bins[:-1] + bins[1:]) * 0.5
        plt.bar(center, hist, width=0.03)
        plt.plot(
            (np.min(data['Steering']),
             np.max(data['Steering'])),
            (samples_per_bin, samples_per_bin)
        )
        plt.title('Data Visualisation')
        plt.xlabel('Steering Angle')
        plt.ylabel('Number of Samples')
        plt.show()

    remove_index_list: list[int] = []
    for x in range(bin_count):
        bin_data_list = []
        for i in range(len(data['Steering'])):
            if data['Steering'][i] >= bins[x] and data['Steering'][i] <= bins[x + 1]:
                bin_data_list.append(i)
        bin_data_list = shuffle(bin_data_list)
        bin_data_list = bin_data_list[samples_per_bin:]
        remove_index_list.extend(bin_data_list)

    print('Removed Images: ', len(remove_index_list))
    data.drop(data.index[remove_index_list], inplace=True)
    print('Remaining Images: ', len(data))
    if display:
        hist, _ = np.histogram(data['Steering'], (bin_count))
        plt.bar(center, hist, width=0.03)
        plt.plot((np.min(data['Steering']),
                  np.max(data['Steering'])),
                 (samples_per_bin, samples_per_bin))
        plt.title('Balanced Data')
        plt.xlabel('Steering Angle')
        plt.ylabel('Number of Samples')
        plt.show()
    return data


# Generate data
def generate_data(
    images_path: [str],
    steering_list: [int],
    batch_size: int,
    train_flag
):
    while True:
        img_batch = []
        steering_batch = []

        for i in range(batch_size):
            index = random.randint(0, len(images_path) - 1)
            if train_flag:
                img, steering = augment_image(
                    images_path[index], steering_list[index])
            else:
                img = mpimg.imread(images_path[index])
                steering = steering_list[index]
            img = pre_process(img)
            img_batch.append(img)
            steering_batch.append(steering)
        yield (np.asarray(img_batch), np.asarray(steering_batch))

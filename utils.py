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

# Preprocessing preparation


def load_data(path: str, data: pd.DataFrame) -> tuple[str, int]:
    images_path: list[str] = []
    steering: list[int] = []
    for i in range(len(data)):
        indexed_data = data.iloc[i]
        images_path.append(os.path.join(path, indexed_data[0]))
        steering.append(float(indexed_data[1]))
    images_path = np.asarray(images_path)
    steering = np.asarray(steering)
    return images_path, steering


# Augment
def augment_image(img_path: str, steering: int):
    img = mpimg.imread(img_path)
    if np.random.rand() < 0.5:
        pan = iaa.Affine(translate_percent={
                         "x": (-0.1, 0.1), "y": (-0.1, 0.1)})
        img = pan.augment_image(img)
    if np.random.rand() < 0.5:
        zoom = iaa.Affine(scale=(1, 1.2))
        img = zoom.augment_image(img)
    if np.random.rand() < 0.5:
        brightness = iaa.Multiply((0.5, 1.2))
        img = brightness.augment_image(img)
    if np.random.rand() < 0.5:
        img = cv2.flip(img, 1)
        steering = -steering
    return img, steering


def pre_process(img):
    img = img[54:120, :, :]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img,  (3, 3), 0)
    img = cv2.resize(img, (200, 66))
    img = img/255
    return img


# Create model
def create_model():
    model = Sequential()

    model.add(Convolution2D(24, (5, 5), (2, 2),
              input_shape=(66, 200, 3), activation='elu'))
    model.add(Convolution2D(36, (5, 5), (2, 2), activation='elu'))
    model.add(Convolution2D(48, (5, 5), (2, 2), activation='elu'))
    model.add(Convolution2D(64, (3, 3), activation='elu'))
    model.add(Convolution2D(64, (3, 3), activation='elu'))

    model.add(Flatten())
    model.add(Dense(100, activation='elu'))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(1))

    model.compile(Adam(lr=0.0001), loss='mse')
    return model


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

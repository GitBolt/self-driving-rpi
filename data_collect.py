import os
import cv2
import time
import pandas as pd


class CollectData:
    def __init__(self) -> None:
        self.folder_count = 0
        self.image_list = []
        self.steering_list = []
        self.collection_directory = os.path.join(os.getcwd(), 'data')

        while os.path.exists(os.path.join(self.collection_directory, f'image{str(self.folder_count)}')):
            self.folder_count += 1

        self.new_path = self.collection_directory + \
            "/image" + str(self.folder_count)

    def save_log(self):
        raw_data = {'Image': self.image_list, 'Steering': self.steering_list}
        df = pd.DataFrame(raw_data)
        df.to_csv(os.path.join(self.collection_directory,
                  f'log_{str(self.folder_count)}'))
        print("Log saved, total images: ", len(self.image_list))

    def save_data(self, img, steering):
        file_name = os.path.join(self.new_path, f'img{time.time()}.jpg')
        cv2.imwrite(file_name, img)
        self.image_list.append(file_name)
        self.steering_list.append(steering)

    def start(self, loops: int) -> None:
        os.makedirs(self.new_path)
        cap = cv2.VideoCapture(0)
        for x in range(loops if loops else 10):
            _, img = cap.read()
            self.save_data(img, 0.5)
            cv2.waitKey(1)
            cv2.imshow("Image", img)
        self.save_log()

# data_collect = CollectData()
# data_collect.start(10)

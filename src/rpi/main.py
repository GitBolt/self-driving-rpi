import cv2
import numpy as np
from tensorflow.keras.models import load_model

import motor as mm

steering_sensitivity = 0.70  # Steering Sensitivity
max_throttle = 0.22  # Throttle (Forward speed percentage)
motor = mm.Motor(2, 3, 4, 17, 22, 27)

model = load_model('model.h5')

def get_image(display=False, size=[480, 240]):
    capture = cv2.VideoCapture(0)
    _, img = capture.read()
    img = cv2.resize(img, (size[0], size[1]))
    if display:
        cv2.imshow('Image Display', img)
    return img


def pre_process(img):
    img = img[54:120, :, :]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.resize(img, (200, 66))
    img = img / 255
    return img


while True:
    img = get_image(False, [240, 120])
    img = np.asarray(img)
    img = pre_process(img)
    img = np.array([img])
    steering = float(model.predict(img))
    print("Steering: ", steering * steering_sensitivity)
    motor.move(maxThrottle, -steering * steering_sensitivity)
    cv2.waitKey(1)

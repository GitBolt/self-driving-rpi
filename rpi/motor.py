import time
import RPi.GPIO as GPIO


GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)


class Motor:
    def __init__(self, enable_pin_1, input_pin_1, input_pin_2, enable_pin_2, input_pin_3, input_pin_4):
        self.enable_pin_1 = enable_pin_1
        self.input_pin_1 = input_pin_1
        self.input_pin_2 = input_pin_2
        self.enable_pin_2 = enable_pin_2
        self.input_pin_3 = input_pin_3
        self.input_pin_4 = input_pin_4
        GPIO.setup(self.enable_pin_1, GPIO.OUT)
        GPIO.setup(self.input_pin_1, GPIO.OUT)
        GPIO.setup(self.input_pin_2, GPIO.OUT)
        GPIO.setup(self.enable_pin_2, GPIO.OUT)
        GPIO.setup(self.input_pin_3, GPIO.OUT)
        GPIO.setup(self.input_pin_4, GPIO.OUT)
        self.pwm_1 = GPIO.PWM(self.enable_pin_1, 100)
        self.pwm_2 = GPIO.PWM(self.enable_pin_2, 100)
        self.pwm_1.start(0)
        self.pwm_2.start(0)

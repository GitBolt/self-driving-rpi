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

    def move(self, speed=0.5, turn=0, t=0):
        speed *= 100
        turn *= 70
        left_speed = speed-turn
        right_speed = speed+turn

        if left_speed > 100:
            left_speed = 100
        elif left_speed < -100:
            left_speed = -100
        if right_speed > 100:
            right_speed = 100
        elif right_speed < -100:
            right_speed = -100
            
        self.pwm_1.ChangeDutyCycle(abs(left_speed))
        self.pwm_2.ChangeDutyCycle(abs(right_speed))
        if left_speed > 0:
            GPIO.output(self.input_pin_1, GPIO.HIGH)
            GPIO.output(self.input_pin_2, GPIO.LOW)
        else:
            GPIO.output(self.input_pin_1, GPIO.LOW)
            GPIO.output(self.input_pin_2, GPIO.HIGH)
        if right_speed > 0:
            GPIO.output(self.input_pin_3, GPIO.HIGH)
            GPIO.output(self.input_pin_4, GPIO.LOW)
        else:
            GPIO.output(self.input_pin_3, GPIO.LOW)
            GPIO.output(self.input_pin_4, GPIO.HIGH)
        sleep(t)

    def stop(self, t=0):
        self.pwm_1.ChangeDutyCycle(0)
        self.pwm_2.ChangeDutyCycle(0)
        sleep(t)


def main():
    motor.move(0.5, 0, 2)
    motor.stop(2)
    motor.move(-0.5, 0, 2)
    motor.stop(2)
    motor.move(0, 0.5, 2)
    motor.stop(2)
    motor.move(0, -0.5, 2)
    motor.stop(2)

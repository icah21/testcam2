# servo.py

import time
import threading
import RPi.GPIO as GPIO

class ServoController:
    def __init__(self, pin=18):
        self.pin = pin
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.pin, GPIO.OUT)
        self.pwm = GPIO.PWM(self.pin, 50)  # 50Hz for SG90
        self.pwm.start(0)
        self.lock = threading.Lock()

    def set_angle(self, angle):
        duty = 2 + (angle + 90) * 10 / 180  # Convert angle to duty cycle
        with self.lock:
            self.pwm.ChangeDutyCycle(duty)
            time.sleep(0.5)
            self.pwm.ChangeDutyCycle(0)  # Stop signal to prevent jitter

    def rotate_to_sort(self, bean_type):
        angle_map = {
            "Criollo": 45,
            "Forastero": 90,
            "Trinitario": -45,
            "Unknown": -90
        }
        angle = angle_map.get(bean_type, 0)
        self.set_angle(angle)
        time.sleep(0.5)
        self.set_angle(0)

    def cleanup(self):
        self.pwm.stop()
        GPIO.cleanup()

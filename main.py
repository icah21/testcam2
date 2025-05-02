# main.py

import threading
from camera import start_camera, get_detected_type
from servo import ServoController
import time

def monitor_and_sort(servo):
    last_type = None
    while True:
        detected = get_detected_type()
        if detected and detected != last_type:
            print(f"Sorting bean type: {detected}")
            servo.rotate_to_sort(detected)
            last_type = detected
        time.sleep(0.5)

if __name__ == "__main__":
    try:
        servo = ServoController()
        threading.Thread(target=start_camera, daemon=True).start()
        monitor_and_sort(servo)
    except KeyboardInterrupt:
        servo.cleanup()

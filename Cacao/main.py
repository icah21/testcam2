# main.py
import time
from camera import get_detected_type
from servo import ServoController


def main():
    print("Starting cacao bean classification system...")
    servo = ServoController()

    try:
        while True:
            bean_type = get_detected_type()
            if bean_type:
                print(f"Detected bean type: {bean_type}")
                servo.rotate_to_sort(bean_type)
            else:
                print("No bean detected or unknown type.")
            time.sleep(2)

    except KeyboardInterrupt:
        print("Program interrupted by user.")
    finally:
        servo.cleanup()
        print("Cleaned up GPIO and exiting.")

if __name__ == "__main__":
    main()

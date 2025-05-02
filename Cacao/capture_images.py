import cv2
import os

# Set class name here (repeat per class)
class_name = "trinitario"
save_dir = f"captured_images/{class_name}"
os.makedirs(save_dir, exist_ok=True)

cap = cv2.VideoCapture(1)
count = 0

print("Press 's' to save image, 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Webcam - Press 's' to Save", frame)
    key = cv2.waitKey(1)

    if key == ord('s'):
        filename = os.path.join(save_dir, f"{class_name}_{count}.jpg")
        cv2.imwrite(filename, frame)
        print(f"Saved {filename}")
        count += 1
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

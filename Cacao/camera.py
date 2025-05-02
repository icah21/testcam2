import cv2
import numpy as np
from roboflow import Roboflow
import tkinter as tk
from PIL import Image, ImageTk
import threading
import time
import os

# Roboflow setup
rf = Roboflow(api_key="f4UBb9Y1BqAaVoiasTC1")
project = rf.workspace("cacaotrain").project("trained-q5iwo")
model = project.version(2).model

# HSV thresholds
criollo_lower = np.array([0, 10, 180])
criollo_upper = np.array([15, 80, 255])
forastero_lower = np.array([130, 50, 50])
forastero_upper = np.array([170, 255, 255])
trinitario_lower = np.array([10, 50, 100])
trinitario_upper = np.array([30, 255, 255])
min_match_threshold = 10.0

# Tkinter UI
root = tk.Tk()
root.title("Cacao Detection Dashboard")
root.geometry('800x600')

def toggle_fullscreen(_=None):
    state = root.attributes('-fullscreen')
    root.attributes('-fullscreen', not state)

root.bind("<F11>", toggle_fullscreen)

# Layout
video_label = tk.Label(root, bd=2, relief="solid")
video_label.grid(row=0, column=0, rowspan=2, sticky="nsew")

dashboard = tk.Frame(root, bg="#2E2E2E", bd=10)
dashboard.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
root.grid_rowconfigure(0, weight=7)
root.grid_columnconfigure(0, weight=7)

# Logo
try:
    script_dir = os.path.dirname(__file__)
    logo_path = os.path.join(script_dir, "cacao.jpg")
    logo_image = Image.open(logo_path)
    logo_image = logo_image.resize((150, 150), Image.Resampling.LANCZOS)
    logo_tk = ImageTk.PhotoImage(logo_image)
    logo_label = tk.Label(dashboard, image=logo_tk, bg="#2E2E2E")
    logo_label.image = logo_tk
    logo_label.pack(pady=(0, 10))
except Exception as err:
    print(f"Logo load failed: {err}")

# Dashboard vars
criollo_var = tk.StringVar()
forastero_var = tk.StringVar()
trinitario_var = tk.StringVar()
unknown_var = tk.StringVar()
detected_type_var = tk.StringVar()

tk.Label(dashboard, text="🧠 Detection Summary", font=("Arial", 16, "bold"), fg="white", bg="#2E2E2E").pack(pady=10)
tk.Label(dashboard, textvariable=criollo_var, font=("Arial", 12), fg="white", bg="#2E2E2E").pack(pady=5)
tk.Label(dashboard, textvariable=forastero_var, font=("Arial", 12), fg="white", bg="#2E2E2E").pack(pady=5)
tk.Label(dashboard, textvariable=trinitario_var, font=("Arial", 12), fg="white", bg="#2E2E2E").pack(pady=5)
tk.Label(dashboard, textvariable=unknown_var, font=("Arial", 12), fg="white", bg="#2E2E2E").pack(pady=5)
tk.Label(dashboard, textvariable=detected_type_var, font=("Arial", 14, "bold"), fg="#00BFFF", bg="#2E2E2E").pack(pady=(10, 0))

def close_app():
    cap.release()
    cv2.destroyAllWindows()
    root.destroy()

tk.Button(dashboard, text="❌ Exit", font=("Arial", 12), command=close_app, bg="#FF6347", fg="white", relief="flat", padx=15, pady=5).pack(pady=20)

counts = {"Criollo": 0, "Forastero": 0, "Trinitario": 0, "Unknown": 0}
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

latest_frame = None
last_pred_time = 0
latest_detected_type = "Unknown"

def get_detected_type():
    return latest_detected_type

def predict_and_update(frame):
    global counts, last_pred_time, latest_detected_type

    if time.time() - last_pred_time < 1.5:
        return
    last_pred_time = time.time()

    image_path = os.path.join(os.path.dirname(__file__), "frame.jpg")
    cv2.imwrite(image_path, frame)

    try:
        predictions = model.predict(image_path, confidence=40, overlap=30).json()
    except Exception as err:
        print(f"Prediction error: {err}")
        return

    for k in counts:
        counts[k] = 0

    for pred in predictions.get("predictions", []):
        x, y, w, h = map(int, [pred['x'], pred['y'], pred['width'], pred['height']])
        x1, y1 = max(x - w // 2, 0), max(y - h // 2, 0)
        x2, y2 = min(x + w // 2, frame.shape[1]), min(y + h // 2, frame.shape[0])
        crop = frame[y1:y2, x1:x2]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

        masks = {
            "Criollo": cv2.inRange(hsv, criollo_lower, criollo_upper),
            "Forastero": cv2.inRange(hsv, forastero_lower, forastero_upper),
            "Trinitario": cv2.inRange(hsv, trinitario_lower, trinitario_upper),
        }

        color_label = "Unknown"
        for name, mask in masks.items():
            ratio = (cv2.countNonZero(mask) / (crop.size / 3)) * 100
            if ratio > min_match_threshold:
                color_label = name
                counts[name] += 1
                break
        else:
            counts["Unknown"] += 1

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{pred['class']} | {color_label}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    criollo_var.set(f"Criollo: {counts['Criollo']}")
    forastero_var.set(f"Forastero: {counts['Forastero']}")
    trinitario_var.set(f"Trinitario: {counts['Trinitario']}")
    unknown_var.set(f"Unknown: {counts['Unknown']}")

    latest_detected_type = max(counts, key=counts.get)
    detected_type_var.set(f"Detected: {latest_detected_type}")

def update_frame():
    global latest_frame
    ret, frame = cap.read()
    if ret:
        latest_frame = frame.copy()
        threading.Thread(target=predict_and_update, args=(frame.copy(),), daemon=True).start()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = ImageTk.PhotoImage(Image.fromarray(rgb))
        video_label.imgtk = img
        video_label.configure(image=img)
    root.after(30, update_frame)

def start_camera():
    update_frame()
    root.mainloop()

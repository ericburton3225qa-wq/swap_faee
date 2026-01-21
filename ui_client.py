import cv2
import requests
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
import threading

# --- Optional virtual camera ---
try:
    import pyvirtualcam
except ImportError as exc:
    pyvirtualcam = None
    PYVIRTUALCAM_IMPORT_ERROR = exc
    VIRTUAL_CAM_PIXEL_FORMAT = None
    NEEDS_RGB_CONVERSION = False
else:
    PYVIRTUALCAM_IMPORT_ERROR = None
    VIRTUAL_CAM_PIXEL_FORMAT = getattr(pyvirtualcam.PixelFormat, "BGR", pyvirtualcam.PixelFormat.RGB)
    NEEDS_RGB_CONVERSION = VIRTUAL_CAM_PIXEL_FORMAT == pyvirtualcam.PixelFormat.RGB

# --- Config ---
SERVER_URL = "http://127.0.0.1:8000/process-frame"
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
CAMERA_INDEX = 0
DEFAULT_FPS = 30

# Global to store selected image path
selected_image_path = None

# --- Camera helper functions ---
def _configure_capture(cap):
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    for _ in range(10):
        ret, frame = cap.read()
        if not ret:
            continue
        h, w = frame.shape[:2]
        if (w, h) == (FRAME_WIDTH, FRAME_HEIGHT):
            return True
    return False

def _capture_fps(cap):
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps != fps or fps <= 1:
        return DEFAULT_FPS
    return fps

def _available_camera_apis():
    apis = [
        getattr(cv2, "CAP_AVFOUNDATION", None),
        getattr(cv2, "CAP_DSHOW", None),
        getattr(cv2, "CAP_MSMF", None),
        cv2.CAP_ANY,
    ]
    return [api for api in apis if api is not None]

def create_capture():
    for api in _available_camera_apis():
        cap = cv2.VideoCapture(CAMERA_INDEX, api)
        if not cap.isOpened():
            cap.release()
            continue
        if _configure_capture(cap):
            return cap
        cap.release()
    raise RuntimeError("Could not configure the camera to stream at 640x480.")

def create_virtual_cam(width, height, fps):
    if pyvirtualcam is None:
        raise ImportError(
            "pyvirtualcam is required to stream to the OBS Virtual Camera. Install it with 'pip install pyvirtualcam'."
        ) from PYVIRTUALCAM_IMPORT_ERROR
    fps_int = max(1, int(round(fps)))
    return pyvirtualcam.Camera(width=width, height=height, fps=fps_int, fmt=VIRTUAL_CAM_PIXEL_FORMAT)

def _prepare_frame_for_virtual_cam(frame):
    frame_for_cam = frame
    h, w = frame_for_cam.shape[:2]
    if (w, h) != (FRAME_WIDTH, FRAME_HEIGHT):
        frame_for_cam = cv2.resize(frame_for_cam, (FRAME_WIDTH, FRAME_HEIGHT))
    if NEEDS_RGB_CONVERSION:
        frame_for_cam = cv2.cvtColor(frame_for_cam, cv2.COLOR_BGR2RGB)
    return frame_for_cam

# --- GUI actions ---
def select_image():
    global selected_image_path
    path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.jpeg")])
    if path:
        selected_image_path = path

        # Immediately upload the image
        try:
            img = cv2.imread(selected_image_path)
            success, buf = cv2.imencode(".jpg", img)
            if success:
                requests.post(
                    SERVER_URL,
                    params={"source": True},
                    files={"file": ("image.jpg", buf.tobytes(), "image/jpeg")},
                )
                messagebox.showinfo("Image Uploaded", f"Selected image uploaded successfully:\n{selected_image_path}")
                print("Image uploaded successfully")
            else:
                messagebox.showwarning("Upload Failed", "Failed to encode image for upload.")
        except Exception as e:
            messagebox.showerror("Upload Error", f"Failed to upload selected image:\n{e}")

        selected_image_path = None  # Clear after upload

def start_processing():
    def run_camera_loop():
        try:
            cap = create_capture()
            capture_fps = _capture_fps(cap)
            virtual_cam = create_virtual_cam(FRAME_WIDTH, FRAME_HEIGHT, capture_fps)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start camera or virtual cam:\n{e}")
            return

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    continue

                # Send frame to server
                success, buffer = cv2.imencode(".jpg", frame)
                if not success:
                    continue
                try:
                    response = requests.post(
                        SERVER_URL,
                        files={"file": ("frame.jpg", buffer.tobytes(), "image/jpeg")},
                    )
                    nparr = np.frombuffer(response.content, np.uint8)
                    processed_frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    if processed_frame is None:
                        continue
                except Exception as e:
                    print("Failed to send frame:", e)
                    continue

                # Show locally and send to virtual cam
                cv2.imshow("Processed Frame", processed_frame)
                frame_for_cam = _prepare_frame_for_virtual_cam(processed_frame)
                virtual_cam.send(frame_for_cam)
                virtual_cam.sleep_until_next_frame()

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            cap.release()
            virtual_cam.close()
            cv2.destroyAllWindows()

    # Run camera loop in a separate thread so GUI doesn't freeze
    threading.Thread(target=run_camera_loop, daemon=True).start()

# --- GUI ---
root = tk.Tk()
root.title("Camera + Image Upload")

select_btn = tk.Button(root, text="Select & Upload Image", command=select_image, width=25)
select_btn.pack(pady=10)

start_btn = tk.Button(root, text="Start Camera", command=start_processing, width=25)
start_btn.pack(pady=10)

root.mainloop()

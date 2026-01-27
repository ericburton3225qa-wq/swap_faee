import sys
import cv2
import requests
import numpy as np
import threading

from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QFileDialog, QMessageBox, QLabel
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QImage, QPixmap

# Optional virtual camera
try:
    import pyvirtualcam
except ImportError as exc:
    pyvirtualcam = None
    PYVIRTUALCAM_IMPORT_ERROR = exc
    VIRTUAL_CAM_PIXEL_FORMAT = None
    NEEDS_RGB_CONVERSION = False
else:
    VIRTUAL_CAM_PIXEL_FORMAT = getattr(pyvirtualcam.PixelFormat, "BGR", pyvirtualcam.PixelFormat.RGB)
    NEEDS_RGB_CONVERSION = VIRTUAL_CAM_PIXEL_FORMAT == pyvirtualcam.PixelFormat.RGB

# --- Config ---
SERVER_URL = "http://127.0.0.1:8000/process-frame"
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
CAMERA_INDEX = 0
DEFAULT_FPS = 30

selected_image_path = None

class CameraApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Camera + Image Upload")
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self.label = QLabel("Camera Feed")
        self.label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.label)

        self.select_btn = QPushButton("Select & Upload Image")
        self.select_btn.clicked.connect(self.select_image)
        self.layout.addWidget(self.select_btn)

        self.start_btn = QPushButton("Start Camera")
        self.start_btn.clicked.connect(self.start_camera)
        self.layout.addWidget(self.start_btn)

        self.cap = None
        self.virtual_cam = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

    # --- Image Upload ---
    def select_image(self):
        global selected_image_path
        path, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Image Files (*.jpg *.png *.jpeg)")
        if not path:
            return

        selected_image_path = path

        try:
            img = cv2.imread(selected_image_path)
            success, buf = cv2.imencode(".jpg", img)
            if success:
                requests.post(
                    SERVER_URL,
                    params={"source": True},
                    files={"file": ("image.jpg", buf.tobytes(), "image/jpeg")}
                )
                QMessageBox.information(self, "Image Uploaded", f"Selected image uploaded successfully:\n{selected_image_path}")
                print("Image uploaded successfully")
            else:
                QMessageBox.warning(self, "Upload Failed", "Failed to encode image for upload.")
        except Exception as e:
            QMessageBox.critical(self, "Upload Error", f"Failed to upload selected image:\n{e}")

        selected_image_path = None

    # --- Camera ---
    def start_camera(self):
        if self.cap is not None:
            return  # Already running
        try:
            self.cap = cv2.VideoCapture(CAMERA_INDEX)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            if not fps or fps <= 1:
                fps = DEFAULT_FPS
            if pyvirtualcam is not None:
                self.virtual_cam = pyvirtualcam.Camera(width=FRAME_WIDTH, height=FRAME_HEIGHT, fps=int(fps), fmt=VIRTUAL_CAM_PIXEL_FORMAT)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start camera or virtual cam:\n{e}")
            return

        self.timer.start(int(1000 / DEFAULT_FPS))  # Update frames

    def update_frame(self):
        if self.cap is None:
            return
        ret, frame = self.cap.read()
        if not ret:
            return

        # Send frame to server
        try:
            success, buf = cv2.imencode(".jpg", frame)
            if success:
                response = requests.post(SERVER_URL, files={"file": ("frame.jpg", buf.tobytes(), "image/jpeg")})
                nparr = np.frombuffer(response.content, np.uint8)
                processed_frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            else:
                processed_frame = frame
        except Exception as e:
            print("Failed to send frame:", e)
            processed_frame = frame

        # Convert to Qt image
        h, w, ch = processed_frame.shape
        bytes_per_line = ch * w
        qt_img = QImage(processed_frame.data, w, h, bytes_per_line, QImage.Format_BGR888)
        self.label.setPixmap(QPixmap.fromImage(qt_img))

        # Virtual cam
        if self.virtual_cam is not None:
            frame_for_cam = processed_frame
            if NEEDS_RGB_CONVERSION:
                frame_for_cam = cv2.cvtColor(frame_for_cam, cv2.COLOR_BGR2RGB)
            self.virtual_cam.send(frame_for_cam)
            self.virtual_cam.sleep_until_next_frame()

    def closeEvent(self, event):
        self.timer.stop()
        if self.cap:
            self.cap.release()
        if self.virtual_cam:
            self.virtual_cam.close()
        super().closeEvent(event)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CameraApp()
    window.show()
    sys.exit(app.exec())

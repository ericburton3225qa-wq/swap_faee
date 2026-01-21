import cv2
import requests
import numpy as np

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

SERVER_URL = "http://127.0.0.1:8000/process-frame"
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
CAMERA_INDEX = 0
DEFAULT_FPS = 30


def _available_camera_apis():
    """Return camera APIs to try, prioritizing platform-native backends."""
    apis = [
        getattr(cv2, "CAP_AVFOUNDATION", None),
        getattr(cv2, "CAP_DSHOW", None),
        getattr(cv2, "CAP_MSMF", None),
        cv2.CAP_ANY,
    ]
    return [api for api in apis if api is not None]


def _configure_capture(cap):
    """Configure the capture object and confirm we actually get 640x480 frames."""
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    for _ in range(10):  # pull a few frames to let the driver settle
        ret, frame = cap.read()
        if not ret:
            continue
        height, width = frame.shape[:2]
        if (width, height) == (FRAME_WIDTH, FRAME_HEIGHT):
            return True
    return False


def create_capture():
    """Open the camera with a backend that honors 640x480 captures."""
    for api in _available_camera_apis():
        cap = cv2.VideoCapture(CAMERA_INDEX, api)
        if not cap.isOpened():
            cap.release()
            continue

        if _configure_capture(cap):
            return cap

        cap.release()

    raise RuntimeError("Could not configure the camera to stream at 640x480.")


def _capture_fps(cap):
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps != fps or fps <= 1:  # handles 0 and NaN
        return DEFAULT_FPS
    return fps


def create_virtual_cam(width, height, fps):
    if pyvirtualcam is None:
        raise ImportError(
            "pyvirtualcam is required to stream to the OBS Virtual Camera. Install it with 'pip install pyvirtualcam'."
        ) from PYVIRTUALCAM_IMPORT_ERROR

    fps_int = max(1, int(round(fps)))
    return pyvirtualcam.Camera(width=width, height=height, fps=fps_int, fmt=VIRTUAL_CAM_PIXEL_FORMAT)


def _prepare_frame_for_virtual_cam(frame):
    frame_for_cam = frame
    height, width = frame_for_cam.shape[:2]
    if (width, height) != (FRAME_WIDTH, FRAME_HEIGHT):
        frame_for_cam = cv2.resize(frame_for_cam, (FRAME_WIDTH, FRAME_HEIGHT))

    if NEEDS_RGB_CONVERSION:
        frame_for_cam = cv2.cvtColor(frame_for_cam, cv2.COLOR_BGR2RGB)

    return frame_for_cam


def main():
    cap = create_capture()
    capture_fps = _capture_fps(cap)

    try:
        virtual_cam = create_virtual_cam(FRAME_WIDTH, FRAME_HEIGHT, capture_fps)
    except Exception:
        cap.release()
        cv2.destroyAllWindows()
        raise

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                break

            # Encode frame as JPEG
            success, buffer = cv2.imencode(".jpg", frame)
            if not success:
                continue

            # Send frame to server
            response = requests.post(
                SERVER_URL,
                files={"file": ("frame.jpg", buffer.tobytes(), "image/jpeg")},
            )

            # Read processed frame
            nparr = np.frombuffer(response.content, np.uint8)
            processed_frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if processed_frame is None:
                continue

            # Display locally
            cv2.imshow("Processed Frame", processed_frame)

            # Push to OBS Virtual Cam
            frame_for_cam = _prepare_frame_for_virtual_cam(processed_frame)
            virtual_cam.send(frame_for_cam)
            virtual_cam.sleep_until_next_frame()

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        virtual_cam.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

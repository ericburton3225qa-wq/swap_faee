import cv2
import insightface
from insightface.app import FaceAnalysis
import os
import threading
import time
import torch
from gfpgan import GFPGANer
import onnxruntime as ort
import pyvirtualcam
from pyvirtualcam import PixelFormat

assert insightface.__version__ >= '0.7'

# Global objects
FACE_ANALYSER = None
FACE_SWAPPER = None
GFPGAN_ENHANCER = None
THREAD_LOCK = threading.Lock()
SOURCE_FACE = None

def get_face_analyser():
    """Get or create face analyser instance (singleton)"""
    global FACE_ANALYSER
    with THREAD_LOCK:
        if FACE_ANALYSER is None:
            # prefer GPU if available in ONNX Runtime or PyTorch
            try:
                providers = onnxruntime.get_available_providers()
            except Exception:
                providers = []
            use_cuda = 'CUDAExecutionProvider' in providers or torch.cuda.is_available()
            ctx = 0 if use_cuda else -1
            FACE_ANALYSER = FaceAnalysis(name='buffalo_l')
            FACE_ANALYSER.prepare(ctx_id=ctx, det_size=(640, 640))
            print(f"FaceAnalysis initialized with ctx_id={ctx} (use_cuda={use_cuda})")
    return FACE_ANALYSER

def get_face_swapper():
    """Get or create face swapper instance (singleton)"""
    global FACE_SWAPPER
    with THREAD_LOCK:
        if FACE_SWAPPER is None:
            FACE_SWAPPER = insightface.model_zoo.get_model(
                './models/inswapper_128.onnx',
                download=True,
                download_zip=True
            )
    return FACE_SWAPPER

def get_gfpgan_enhancer():
    """Get or create GFPGAN enhancer instance (singleton)"""
    global GFPGAN_ENHANCER
    with THREAD_LOCK:
        if GFPGAN_ENHANCER is None:
            model_path = os.path.join(os.path.dirname(__file__), "models", "GFPGANv1.4.pth")
            if not os.path.exists(model_path):
                raise RuntimeError(f"GFPGAN model not found at: {model_path}")
            
            # Determine device
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
            GFPGAN_ENHANCER = GFPGANer(
                
                model_path=model_path,
                upscale=2,
                arch='clean',
                channel_multiplier=2,
                bg_upsampler=None,
                device=device
            )
    return GFPGAN_ENHANCER

def load_source_face(source_path):
    """Load and extract face from source image"""
    global SOURCE_FACE
    
    if not os.path.exists(source_path):
        raise RuntimeError(f"Source image not found: {source_path}")
    
    src_img = cv2.imread(source_path)
    if src_img is None:
        raise RuntimeError(f"Failed to read source image: {source_path}")
    
    app = get_face_analyser()
    src_faces = app.get(src_img)
    
    if len(src_faces) == 0:
        raise RuntimeError(f"No faces found in source image: {source_path}")
    
    SOURCE_FACE = src_faces[0]
    print(f"✓ Source face loaded from: {source_path}")

def swap_faces_in_frame(frame, enhance=False):
    """Swap source face into all faces detected in the frame"""
    global SOURCE_FACE
    
    if SOURCE_FACE is None:
        return frame
    
    app = get_face_analyser()
    swapper = get_face_swapper()
    
    # Detect faces in current frame
    frame_faces = app.get(frame)
    
    if len(frame_faces) == 0:
        return frame
    
    # Swap source face into each detected face
    result = frame.copy()
    for face in frame_faces:
        result = swapper.get(result, face, SOURCE_FACE, paste_back=True)
    
    # Apply GFPGAN enhancement if enabled
    if enhance:
        try:
            enhancer = get_gfpgan_enhancer()
            cropped_faces, restored_faces, restored_img = enhancer.enhance(result, has_aligned=False, only_center_face=False, paste_back=True)
            result = restored_img
        except Exception as e:
            print(f"Warning: GFPGAN enhancement failed: {str(e)}")
            # Continue without enhancement
    
    return result

# =====================
# Virtual Camera Loop
# =====================
def run_virtual_camera(
    source_img,
    cam_id=0,
    width=640,
    height=480,
    fps=30,
    enhance=False
):
    load_source_face(source_img)

    cap = cv2.VideoCapture(cam_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    if not cap.isOpened():
        raise RuntimeError("Camera not accessible")

    print("[OK] Physical camera opened")

    with pyvirtualcam.Camera(
        width=width,
        height=height,
        fps=fps,
        fmt=PixelFormat.BGR
    ) as vcam:

        print(f"[OK] Virtual Camera started: {vcam.device}")
        print("Use this camera in Zoom / Discord / Browser")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            output = swap_faces_in_frame(frame, enhance=enhance)
            output_resized = cv2.resize(
                output,
                (width, height),          # (width, height) order for cv2.resize
                interpolation=cv2.INTER_LINEAR      # or INTER_AREA for downscaling
            )
            vcam.send(output_resized)
            vcam.sleep_until_next_frame()

    cap.release()

# =====================
# Entry
# =====================
if __name__ == "__main__":
    run_virtual_camera(
        source_img="./1.jpg",   # CHANGE THIS
        cam_id=0,
        width=640,
        height=480,
        fps=30,
        enhance=False
    )

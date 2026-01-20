import cv2
import insightface
from insightface.app import FaceAnalysis
import argparse
import os
import threading
import time
import torch
from gfpgan import GFPGANer
import onnxruntime

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
                'inswapper_128.onnx',
                download=True,
                download_zip=True
            )
    return FACE_SWAPPER

def get_gfpgan_enhancer():
    """Get or create GFPGAN enhancer instance (singleton)"""
    global GFPGAN_ENHANCER
    with THREAD_LOCK:
        if GFPGAN_ENHANCER is None:
            model_path = os.path.join(os.path.dirname(__file__), "Deep-Live-Cam", "models", "GFPGANv1.4.pth")
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

def run_live_swap(source_path, camera_id=0, mirror=False, width=640, height=480, fps=30, enhance=False):
    """Run live camera face swapping with display"""
    
    # Load source face
    print("Loading source face...")
    load_source_face(source_path)
    
    # Load enhancer if requested
    if enhance:
        print("Loading GFPGAN enhancer...")
        try:
            get_gfpgan_enhancer()
            print("✓ GFPGAN enhancer loaded")
        except Exception as e:
            print(f"Warning: Could not load GFPGAN enhancer: {str(e)}")
            print("Continuing without enhancement")
            enhance = False
    
    # Initialize camera
    print(f"Opening camera {camera_id}...")
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera {camera_id}")
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    
    print(f"✓ Camera opened successfully")
    print(f"Resolution: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    print(f"FPS: {cap.get(cv2.CAP_PROP_FPS):.1f}")
    if enhance:
        print("Enhancement: GFPGAN enabled")
    print("\nPress 'q' to quit\n")
    
    frame_count = 0
    start_time = time.time()
    display_available = True
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("Error: Failed to read frame")
                break
            
            # Mirror frame if requested
            if mirror:
                frame = cv2.flip(frame, 1)
            
            # Process frame - swap faces and optionally enhance
            processed = swap_faces_in_frame(frame, enhance=enhance)
            
            # Calculate FPS
            frame_count += 1
            elapsed = time.time() - start_time
            current_fps = frame_count / elapsed if elapsed > 0 else 0
            
            # Add info to display
            display_frame = processed.copy()
            cv2.putText(display_frame, f"FPS: {current_fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(display_frame, f"Frame: {frame_count}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            
            # Add quit instruction
            cv2.putText(display_frame, "Press 'q' to quit", (10, height-20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            # Try to display frame
            if display_available:
                try:
                    cv2.imshow('Face Swapper - Live Camera', display_frame)
                except cv2.error as e:
                    if frame_count == 1:
                        print(f"Note: Display not available (OpenCV error)")
                        print(f"Processing frames in background...\n")
                    display_available = False
            
            # Handle keyboard input
            try:
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    print("\nQuitting...")
                    break
            except KeyboardInterrupt:
                print("\nInterrupted by user")
                break
            except Exception as e:
                # Ignore keyboard handling errors in headless mode
                pass
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        cap.release()
        try:
            cv2.destroyAllWindows()
        except:
            pass
        
        print(f"\n✓ Camera closed")
        print(f"Total frames processed: {frame_count}")
        if elapsed > 0:
            print(f"Total time: {elapsed:.2f}s")
            print(f"Average FPS: {frame_count/elapsed:.1f}")

def main():
    parser = argparse.ArgumentParser(description='Live camera face swapping with display')
    parser.add_argument('-s', '--source', required=True, help='Path to source image (face to swap from)')
    parser.add_argument('-c', '--camera', type=int, default=0, help='Camera ID (default: 0)')
    parser.add_argument('-m', '--mirror', action='store_true', help='Mirror the camera feed')
    parser.add_argument('-e', '--enhance', action='store_true', help='Enable GFPGAN face enhancement')
    parser.add_argument('-w', '--width', type=int, default=640, help='Frame width (default: 640)')
    parser.add_argument('-H', '--height', type=int, default=480, help='Frame height (default: 480)')
    parser.add_argument('-f', '--fps', type=int, default=30, help='Target FPS (default: 30)')
    
    args = parser.parse_args()
    
    try:
        run_live_swap(args.source, camera_id=args.camera, mirror=args.mirror, 
                     width=args.width, height=args.height, fps=args.fps, enhance=args.enhance)
        print("Face swapping completed successfully!")
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())

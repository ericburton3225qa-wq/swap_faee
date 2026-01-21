from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
import cv2
import numpy as np
import io
from face_swap import swap_faces_in_frame, load_source_face_from_array

app = FastAPI()

source_face_loaded = False  # Tracks whether we've loaded the source face

@app.post("/process-frame")
async def process_frame(file: UploadFile = File(...), source: bool = False):
    """
    file: uploaded image (either source face or frame)
    source: if True, treat this file as the new source face
    """
    global source_face_loaded

    # Read uploaded bytes and decode to OpenCV image
    data = await file.read()
    nparr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return {"error": "Failed to decode image"}

    # If source=True, load it as the new source face
    if source or not source_face_loaded:
        try:
            load_source_face_from_array(img)
            source_face_loaded = True
            return {"message": "Source face updated successfully."}
        except Exception as e:
            return {"error": str(e)}

    # Otherwise, treat as a frame and swap faces
    try:
        processed_frame = swap_faces_in_frame(img, enhance=False)
    except Exception as e:
        return {"error": f"Face swap failed: {str(e)}"}

    # Encode back to JPEG
    _, buffer = cv2.imencode(".jpg", processed_frame)
    return StreamingResponse(io.BytesIO(buffer.tobytes()), media_type="image/jpeg")



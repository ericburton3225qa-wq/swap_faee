from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
import cv2
import numpy as np
from face_swap import swap_faces_in_frame, load_source_face
import io
 
app = FastAPI()
load_source_face("./3.jpg")
 
@app.post("/process-frame")
async def process_frame(file: UploadFile = File(...)):
    # Read bytes and decode to image
    data = await file.read()
    nparr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
 
    # Process frame
    img = swap_faces_in_frame(img, enhance=False)
 
    # Encode back to JPEG
    _, buffer = cv2.imencode(".jpg", img)
    return StreamingResponse(io.BytesIO(buffer.tobytes()), media_type="image/jpeg")
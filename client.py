import cv2

import requests

import numpy as np
 
SERVER_URL = "http://127.0.0.1:8000/process-frame"
 
cap = cv2.VideoCapture(0)  # capture from webcam
 
while True:

    ret, frame = cap.read()

    if not ret:

        break
 
    # Encode frame as JPEG

    _, buffer = cv2.imencode(".jpg", frame)

    # Send frame to server

    response = requests.post(SERVER_URL, files={"file": ("frame.jpg", buffer.tobytes(), "image/jpeg")})

    # Read processed frame

    nparr = np.frombuffer(response.content, np.uint8)

    processed_frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
 
    # Display

    cv2.imshow("Processed Frame", processed_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):

        break
 
cap.release()

cv2.destroyAllWindows()

 
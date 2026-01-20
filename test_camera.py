import cv2

# Open the default camera
cam = cv2.VideoCapture(0)

if not cam.isOpened():
    print("Cannot open camera")
    exit()

while True:
    ret, frame = cam.read()
    if not ret:
        print("Can't receive frame")
        break

    # Display the frame
    cv2.imshow("Camera", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cam.release()
cv2.destroyAllWindows()
# import cv2
# print(cv2.__version__)
# print(cv2.getBuildInformation())
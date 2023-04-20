import cv2

INTERVAL = 2000     # [msec.]

cv2.namedWindow("Detected Objects", cv2.WINDOW_FULLSCREEN)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

while True:
    cap.grab()  # Discard frame buffer
    _, frame = cap.read()
    cv2.imshow("Detected Objects", frame)

    cv2.waitKey(2000)

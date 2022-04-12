import cv2 as cv
import mediapipe as mp

webcam = cv.VideoCapture(0)

mpHand = mp.solutions.hands
hands = mpHand.Hands()   

mpDraw = mp.solutions.drawing_utils

while webcam.isOpened():
    frame_status, frame = webcam.read()

    handSearch = hands.process(frame).multi_hand_landmarks

    if handSearch:
        for hand in handSearch:
            for lm in hand.landmark:
                h, w, _ = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
            mpDraw.draw_landmarks(frame, hand, mpHand.HAND_CONNECTIONS)

    
    cv.imshow("Camera", frame)
    if cv.waitKey(1) & 0xFF == 27:
        break

webcam.release()
cv.destroyAllWindows()
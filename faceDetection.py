import cv2 
import mediapipe as mp

mpFace = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils

webcam = cv2.VideoCapture(0)

with mpFace.FaceDetection(model_selection=0, min_detection_confidence=0.5) as faces:
  while webcam.isOpened():
    frame_status, frame = webcam.read()

    if not frame_status:
      print("Camera failed.")
      break

    frame.flags.writeable = False
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    detectedFaces = faces.process(frame)

    frame.flags.writeable = True
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    if detectedFaces.detections:
      for face in detectedFaces.detections:
        mpDraw.draw_detection(frame, face)

    cv2.imshow('MediaPipe Face Detection', frame)

    if cv2.waitKey(5) & 0xFF == 27:
      break

webcam.release()
cv2.destroyAllWindows()
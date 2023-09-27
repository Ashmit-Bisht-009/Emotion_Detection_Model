import cv2
from deepface import DeepFace

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    cap=cv2.VideoCapture(1)
if not cap.isOpened():
    raise IOError("Cannot open webcam")

while True:
    ret, frame = cap.read()
    result = DeepFace.analyze(frame, enforce_detection=False, actions=['emotion'])
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, result['dominant_emotion'], (x, y - 20), font, 0.5, (0, 0, 255), 2, cv2.LINE_4)
        if result['dominant_emotion'] == 'happy':
            cv2.putText(frame, "You look happy today!", (x, y - 50), font, 0.5, (0, 0, 255), 2, cv2.LINE_4)
        elif result['dominant_emotion'] == 'neutral':
            cv2.putText(frame, "You look neutral today!", (x, y - 50), font, 0.5, (0, 0, 255), 2, cv2.LINE_4)
        elif result['dominant_emotion'] == 'angry':
            cv2.putText(frame, "You look angry today!", (x, y - 50), font, 0.5, (0, 0, 255), 2, cv2.LINE_4)
        elif result['dominant_emotion'] == 'sad':
            cv2.putText(frame, "You look sad today!", (x, y - 50), font, 0.5, (0, 0, 255), 2, cv2.LINE_4)
        else:
            cv2.putText(frame, "I'm not sure how you're feeling today!", (x, y - 50), font, 0.5, (0, 0, 255), 2, cv2.LINE_4)

    cv2.imshow('Original video', frame)
    if cv2.waitKey(2) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

import threading
import cv2
import numpy as np
from deepface import DeepFace
import os

cap = cv2.VideoCapture(1, cv2.CAP_AVFOUNDATION)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

employee_db = '/Users/mohamamdmustafa/Documents/Personal Projects/AI projects/Group recognition/employees'
threshold = 0.55
count = 0
face_results = []
threadRunning = False
lock = threading.Lock()

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def check_all_faces(frame):
    global face_results, threadRunning

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detected_faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))

    results = []

    for (x, y, w, h) in detected_faces:
        face_crop = frame[y:y+h, x:x+w]

        try:
            df = DeepFace.find(img_path=face_crop, db_path=employee_db, enforce_detection=False, silent=True)

            if len(df) > 0 and not df[0].empty:
                top_match = df[0].iloc[0]
                distance = top_match['distance']

                if distance < threshold:
                    filename = os.path.basename(top_match['identity'])
                    name = os.path.splitext(filename)[0].replace('_', ' ').title()
                    results.append((x, y, w, h, name, True))
                
                else:
                    results.append((x, y, w, h, "Unknown", False))
            
            else:
                results.append((x, y, w, h, "Unknown", False))
        
        except Exception as e:
            results.append((x, y, w, h, "Unknown", False))
    
    with lock:
        face_results = results

    threadRunning = False

while True:
    ret, frame = cap.read()

    if not ret:
        break

    if count % 30 == 0 and not threadRunning:
        threadRunning = True
        threading.Thread(target=check_all_faces, args=(frame.copy(),), daemon=True).start()

    count += 1

    with lock:
        current_results = list(face_results)

        for (x, y, w, h, name, is_employee) in current_results:
            color = (0, 255, 0) if is_employee else (0, 0, 255)
            label = name if is_employee else f"Not Employee: {name}"

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow("Group Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
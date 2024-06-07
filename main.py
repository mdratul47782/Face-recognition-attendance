import face_recognition
import cv2
import numpy as np
import csv
from datetime import datetime
import os

def load_and_encode_faces(face_directory):
    known_face_encodings = []
    known_face_names = []

    for filename in os.listdir(face_directory):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            name = os.path.splitext(filename)[0]
            image_path = os.path.join(face_directory, filename)
            image = face_recognition.load_image_file(image_path)
            encoding = face_recognition.face_encodings(image)[0]
            known_face_encodings.append(encoding)
            known_face_names.append(name)
    
    return known_face_encodings, known_face_names

face_directory = "faces"
known_face_encodings, known_face_names = load_and_encode_faces(face_directory)

recognized_names = set()
now = datetime.now()
current_date = now.strftime("%Y-%m-%d")
f = open(f"{current_date}.csv", "w+", newline="")
lnwriter = csv.writer(f)

video_capture = cv2.VideoCapture(0)

print("Starting video capture...")
while True:
    ret, frame = video_capture.read()
    if not ret:
        print("Failed to capture frame")
        break

    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        face_distance = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distance)

        if matches[best_match_index]:
            name = known_face_names[best_match_index]

            if name not in recognized_names:
                recognized_names.add(name)
                now = datetime.now()
                current_date_time = now.strftime("%Y-%m-%d %H:%M:%S")
                lnwriter.writerow([name, current_date_time])
                print(f"Recognized {name} at {current_date_time}")

    cv2.imshow("Attendance", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("Exiting...")
        break

f.close()
video_capture.release()
cv2.destroyAllWindows()
print("Video capture ended and CSV file closed.")

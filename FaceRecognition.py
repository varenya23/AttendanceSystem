import face_recognition
import cv2
import numpy as np
import csv
import os
from datetime import datetime

video_capture = cv2.VideoCapture(0)

jobs_img = face_recognition.load_image_file("photos/jobs.jpeg")
jobs_encoding = face_recognition.face_encodings(jobs_img)[0]

monalisa_img = face_recognition.load_image_file("photos/monalisa.jpeg")
monalisa_encoding = face_recognition.face_encodings(monalisa_img)[0]

dan_img = face_recognition.load_image_file("photos/dan.jpeg")
dan_encoding = face_recognition.face_encodings(dan_img)[0]

tom_img = face_recognition.load_image_file("photos/tom.jpeg")
tom_encoding = face_recognition.face_encodings(tom_img)[0]

known_face_encoding = [
    jobs_encoding, monalisa_encoding, dan_encoding, tom_encoding
]

known_face_names = [
    "jobs", "monalisa", "dan", "tom"
]

students = known_face_names.copy()

face_locations = []
face_encodings = []
face_names = []

now = datetime.now()
current_date = now.strftime("%Y-%m-%d")

f = open(current_date+'.csv','w+',newline = '')
lnwriter = csv.writer(f)

while True:
    _,frame=video_capture.read()
    small_frame=cv2.resize(frame,(0,0),fx=0.25,fy=0.25)
    rgb_small_frame=small_frame[:,:,::-1]
    if True:
        face_locations=face_recognition.face_location(rgb_small_frame)
        face_encodings=face_recognition.face_encodings(rgb_small_frame,face_locations)
        face_names=[]
        for face_encoding in face_encodings:
            matches=face_recognition.compare_faces(known_face_encoding,face_encoding)
            name=""
            face_distance=face_recognition.face_distance(known_face_encoding,face_encoding)
            best_match_index=np.argmin(face_distance)
            if matches[best_match_index]:
                name=known_face_names[best_match_index]
            face_names.append(name)
            if name in known_face_names:
                if name in students:
                    students.remove(name)
                    print(students)
                    current_time=now.strftime("%H-%M-%S")
                    lnwriter.writerow([name,current_time])
    cv2.imshow("attendence system",frame)
    if cv2.waitkey(1) & 0xFF==ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
f.close()

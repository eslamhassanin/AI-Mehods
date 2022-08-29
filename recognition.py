import face_recognition
import cv2
import numpy as np
import glob
import pickle
from deepface import DeepFace
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

net = cv2.dnn.readNet('/Users/feelingsalwaysdarker/PycharmProjects/pythonProject2/venv/yolov3.weights','/Users/feelingsalwaysdarker/PycharmProjects/pythonProject2/venv/yolov3.cfg')
classes = []
with open('/Users/feelingsalwaysdarker/PycharmProjects/pythonProject2/venv/coco.names','r') as f:
	classes = f.read().splitlines()
video_capture = cv2.VideoCapture(0)

f=open("ref_name.pkl","rb")
ref_dictt=pickle.load(f)
f.close()
f=open("ref_embed.pkl","rb")
embed_dictt=pickle.load(f)
f.close()

known_face_encodings = []
known_face_names = []

for name , embed_list in embed_dictt.items():
    for my_embed in embed_list:
        known_face_encodings +=[my_embed]
        known_face_names += [name]
font_scale=3
video_capture = cv2.VideoCapture(0)
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True  :

    ـ, frame = video_capture.read()

    try:
        height, width, _ = frame.shape
    except AttributeError as e:
        continue

    blob = cv2.dnn.blobFromImage(frame, 1 / 255, (416, 416), (0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    output_Layers_names = net.getUnconnectedOutLayersNames()
    layerOutput = net.forward(output_Layers_names)
    boxes = []
    confidences = []
    classes_ids = []

    for output in layerOutput:
        for detection in output:
            scores = detection[5:]
            classes_id = np.argmax(scores)
            confidence = scores[classes_id]
            if confidence > 0.4:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                classes_ids.append(classes_id)
                indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.9)
                font = cv2.FONT_HERSHEY_DUPLEX
                colors = np.random.uniform(0, 255, size=(len(boxes), 3))

                for i in indexes.flatten():
                    x, y, w, h = boxes[i]
                    label = str(classes[classes_ids[i]])
                    confidence = str(round(confidences[i], 2))
                    color = colors[i]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(frame, label + " " + confidence, (x, y + 20), font, 2, (255, 255, 255), 2)

    result = DeepFace.analyze(frame, actions=['emotion'],enforce_detection=False)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray,1.1,4)



    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]
    if process_this_frame:
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
            face_names.append(name)
    process_this_frame = not process_this_frame
    for (topScreen, rightScreen, bottomScreen, leftScreen), name in zip(face_locations, face_names):
        for (x, y, w, h), faces in zip(face_locations, face_names):
         x *= 4
         y *= 4
         w += 4
         h *= 4
         cv2.rectangle(frame, (h, x), (y, w), (255, 55, 255))
         cv2.rectangle(frame, (h, w - 35), (y, w), (0, 0, 255))
         font1 = cv2.FONT_HERSHEY_DUPLEX
        topScreen *= 4
        rightScreen *= 4
        bottomScreen *= 4
        leftScreen *= 4
        cv2.rectangle(frame, (leftScreen, topScreen), (rightScreen, bottomScreen), (0, 0, 255), 2)
        cv2.rectangle(frame, (leftScreen, bottomScreen - 35), (rightScreen, bottomScreen), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, result['dominant_emotion'], (h + 6, w - 6), font1, 3, (0, 0, 255), 1)
        cv2.putText(frame, name, (leftScreen + 6, bottomScreen - 6), font, 1.0, (255, 255, 255), 1)
        font2 = cv2.FONT_HERSHEY_DUPLEX






        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
video_capture.release()
cv2.destroyAllWindows()
import pyrebase
from flask import *
import face_recognition as fr
import cv2
from face_verification.face_verification import FaceVerification
from tensorflow.keras.preprocessing.image import img_to_array
from utils.logger import resultLogger
from tensorflow.keras.models import load_model
from datetime import datetime
import os
from utils.utils import configInfo
import numpy as np
import time

config = configInfo("config/config.json")
config = config["firebase_config"]

# firebase config of app initialize
firebase = pyrebase.initialize_app(config)
storage = firebase.storage()
db = firebase.database()

def face_gatherer(configfile, updated_phone_number):
#############################################################
    config = configInfo(configfile)
    hyperparameters = config["hyperparameters"]
    width, height, _ = hyperparameters["size"]
    model = load_model(config["best_saved_model"])
    le = config["le"]["classes"]
##############################################################

    fv = FaceVerification(configfile)
    known_face_encodings, known_face_names, face_locations, face_encodings, face_names, process_this_frame = fv.face_information()

    filename = updated_phone_number + ".mp4"
    filepath = os.path.join("video", "Register", filename)
    video_capture = cv2.VideoCapture(filepath)

    w, h = video_capture.get(3), video_capture.get(4)

    if h == w:
        angle = cv2.ROTATE_90_CLOCKWISE
    else:
        angle = cv2.ROTATE_90_COUNTERCLOCKWISE

    if updated_phone_number not in os.listdir("image"):
        os.mkdir(os.path.join("image", updated_phone_number))

    i = 0

    while True:
        ret, frame = video_capture.read()
        frame = cv2.rotate(frame, angle)
        if ret == False:
            break

        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]  # BGR을 RGB로 변환

        if process_this_frame:
            face_locations = fr.face_locations(rgb_small_frame)

        process_this_frame = not process_this_frame

        for (top, right, bottom, left) in face_locations:
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

###############################################
            face = frame[top:bottom, left:right]
            size = face.shape
            face_image = face
            if i % 10 == 0:
                cv2.imwrite(f"image/{updated_phone_number}/test_{i}.jpg", face_image)
        i += 1


def face_liveness_detector(updated_phone_number):
#############################################################
    config = configInfo("config/config.json")
    hyperparameters = config["hyperparameters"]
    width, height, _ = hyperparameters["size"]
    model = load_model(config["best_saved_model"])
    le = config["le"]["classes"]
##############################################################

    fv = FaceVerification("config/config.json")
    known_face_encodings, known_face_names, face_locations, face_encodings, face_names, process_this_frame = fv.face_information()

    filename = updated_phone_number + ".mp4"
    filepath = os.path.join("video", "Login", filename)
    video_capture = cv2.VideoCapture(filepath)
    w, h = video_capture.get(3), video_capture.get(4)

    if h == w:
        angle = cv2.ROTATE_90_CLOCKWISE
    else:
        angle = cv2.ROTATE_90_COUNTERCLOCKWISE

    # save test images per frame
    now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    # os.mkdir(os.path.join("image", "frame", now))
    logger = resultLogger(os.path.join(config["logpath"], ("logs_" + now)) + ".log")
    i = 0

    while True:
        ret, frame = video_capture.read()
        # frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        frame = cv2.rotate(frame, angle)
        if ret == False:
            break

        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]  # BGR을 RGB로 변환

        if process_this_frame:
            face_locations = fr.face_locations(
                rgb_small_frame)  # Returns an array of bounding boxes of human faces in a image
            face_encodings = fr.face_encodings(rgb_small_frame,
                                               face_locations)  # Given an image, return the 128-dimension face encoding for each face in the image.

            face_names = []
            for face_encoding in face_encodings:
                matches = fr.compare_faces(known_face_encodings,
                                           face_encoding, tolerance=0.43)  # Compare a list of face encodings against a candidate encoding to see if they match.
                # tolerance: How much distance between faces to consider it a match. Lower is more strict. 0.6 is typical best performance.
                name = "Unknown"

                # Given a list of face encodings, compare them to a known face encoding and get a euclidean distance
                # for each comparison face. The distance tells you how similar the faces are.
                face_distances = fr.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)

                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

                face_names.append(name)

        process_this_frame = not process_this_frame

        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

###############################################
            face = frame[top:bottom, left:right]
            size = face.shape
            face = cv2.resize(face, (width, height))
            # cv2.imwrite("test.jpg", face)
            # face = face.astype("float") / 255.0
            face = img_to_array(face)
            face = np.expand_dims(face, axis=0)

            preds = model.predict(face)[0]
            j = np.argmax(preds)
            label = le[j]
###############################################

            logger.info(f"{name} {label} {max(preds)} {size[0]} {size[1]} {size[2]}")

    logfile = os.path.join(config["logpath"], ("logs_" + now)) + ".log"

    return logfile, updated_phone_number

def logTest(config, updated_phone_number):
    with open(config) as f:
        logs = f.readlines()

    total = len(logs)
    rkc, ruc, fkc, fuc, cnt = 0, 0, 0, 0, 0

    for log in logs:
        log = log.strip()
        name, label, prob, width, height, channel = log.split()[2:]

        if name != "Unknown" and label == "real":
            rkc += 1
        elif name == "Unknown" and label == "real":
            ruc += 1
        elif name != "Unknown" and label == "fake":
            fkc += 1
        else:
            fuc += 1

        if name == updated_phone_number:
            cnt += 1

    ratio = rkc / total
    correct = cnt / total

    if round(ratio, 2) >= 0.65 and correct >= 0.5:
        return "accept"

    return "denied"


def stream_handler(message):
    stream_handler.var = None
    print(message["event"]) # put
    print(message["path"]) # /-K7yGTTEp7O549EzTYtI
    print(message["data"]) # {'title': 'Pyrebase', "body": "etc..."}

    #

    try:
        if message["data"].startswith("com.google.android.gms.tasks.zzu@"):
            updated_phone_number = message['path'].split('/')[1]
            state = message["path"].split('/')[2]

            print(updated_phone_number, state)

            if state == "loginUrl":
                storage.child(f"Login/{updated_phone_number}").download(f"./video/Login/{updated_phone_number}.mp4")
                print("Login video Downloaded")
                logfile, updated_phone_number = face_liveness_detector(updated_phone_number)
                result = logTest(logfile, updated_phone_number)
                print(result)
                f = open("config/result.txt", 'w')
                f.write(result)
                stream_handler.var = result



            elif state == "url":
                storage.child(f"Register/{updated_phone_number}").download(f"./video/Register/{updated_phone_number}.mp4")
                print("Register video Downloaded")
                face_gatherer("config/config.json", updated_phone_number)
                print("Face Cropped")

    except:
        pass

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def main():
    my_stream = db.child("UserList").stream(stream_handler)

    return "OK"


@app.route('/mobile', methods=['GET','POST'])
def mobile():
    # time.sleep(30)
    # return stream_handler.var
    while True:
        if stream_handler.var:
            result = stream_handler.var
            break
        else:
            continue
    stream_handler.var = None
    return result


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
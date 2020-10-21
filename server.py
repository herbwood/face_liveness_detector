import pyrebase
from flask import *
import face_recognition as fr
import cv2
import numpy as np
from face_verification.face_verification import FaceVerification
from utils.utils import configInfo
from tensorflow.keras.models import load_model
import os

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
    video_capture = cv2.VideoCapture(os.path.join("video", filename))

    os.mkdir(os.path.join("image", updated_phone_number))
    i = 0

    while True:
        ret, frame = video_capture.read()
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        if ret == False:
            break

        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]  # BGR을 RGB로 변환

        if process_this_frame:
            face_locations = fr.face_locations(
                rgb_small_frame)
            face_encodings = fr.face_encodings(rgb_small_frame,
                                               face_locations)

            face_names = []
            for face_encoding in face_encodings:
                matches = fr.compare_faces(known_face_encodings,
                                           face_encoding, tolerance=0.43)
                name = "Unknown"
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
            face_image = face
            face = cv2.resize(face, (width, height))
            cv2.imwrite(f"image/{updated_phone_number}/test_{i}.jpg", face_image)
        i += 1


def stream_handler(message):
    print(message["event"]) # put
    print(message["path"]) # /-K7yGTTEp7O549EzTYtI
    print(message["data"]) # {'title': 'Pyrebase', "body": "etc..."}
    state = "Register"
    try:
        state = message["data"]["state"]
    except:
        pass

    try:
        if message["data"].startswith("com.google.android.gms.tasks.zzu@"):
            updated_phone_number = message['path'].split('/')[1]
            storage.child(f"{state}/{updated_phone_number}").download(f"./video/{updated_phone_number}.mp4")
            print("Downloaded")
            face_gatherer("config/config.json", updated_phone_number)
            print("Face Cropped")

    except:
        pass


app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def main():
    my_stream = db.child("UserList").stream(stream_handler)

if __name__ == "__main__":
    app.run(debug=True)
import face_recognition as fr
import os
from utils.utils import configInfo

class FaceVerification:

    def __init__(self, config):
        self.config = configInfo(config)

    def face_information(self):
        known_face_encodings = []
        known_face_names = []
        face_locations = []
        face_encodings = []
        face_names = []
        process_this_frame = True

        image_path = self.config["face_verification"]
        # image_path = "../image"
        for dir in os.listdir(image_path):
            if dir == "frame" or dir == ".gitkeep":
                continue
            for image in os.listdir(os.path.join(image_path, dir)):
                name = os.path.splitext(os.path.basename(os.path.join(image_path, dir, image)))[0]

                loaded_image = fr.load_image_file(os.path.join(image_path, dir, image))
                loaded_face_encodings = fr.face_encodings(loaded_image)[0]
                known_face_encodings.append(loaded_face_encodings)
                known_face_names.append(dir)

        return known_face_encodings, known_face_names, face_locations, face_encodings, face_names, process_this_frame


if __name__ == "__main__":
    fv = FaceVerification("../config/config.json")
    a, b, d, e, f, g = fv.face_information()
    print(b)



import shutil
import face_recognition as fr
import cv2
from utils.utils import configInfo
import os

def gather_example(config, saved_video_path, image_save_path):
    config = configInfo(config)

    video_capture = cv2.VideoCapture(saved_video_path)
    dirname = os.path.basename(saved_video_path).split(".")[0]

    if dirname not in os.listdir(image_save_path):
        os.mkdir(f"{image_save_path}/{dirname}")

    i = 0
    process_this_frame = True

    while True:
        ret, frame = video_capture.read()
        if ret == False:
            break

        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]  # BGR을 RGB로 변환

        if process_this_frame:
            face_locations = fr.face_locations(
                rgb_small_frame)  # Returns an array of bounding boxes of human faces in a image
            face_encodings = fr.face_encodings(rgb_small_frame,
                                               face_locations)  # Given an image, return the 128-dimension face encoding for each face in the image.

        for (top, right, bottom, left) in face_locations:
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            face = frame[top:bottom, left:right]
            if i % 10 == 0:
                cv2.imwrite(f"{image_save_path}/{dirname}/{i}.jpg", face)
                print(f"{dirname}_{i}.jpg saved")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        else:
            i += 1

    video_capture.release()
    cv2.destroyAllWindows()


def extract_image_name(basepath, file):
    with open(file) as f:
        extracted = f.readlines()

    extracted = [os.path.join(basepath, data.split(' ')[0].split('\\')[0], data.split(' ')[0].split('\\')[1]) for data
                 in extracted]

    return extracted


def copy_images_to_dir(imagepathlist, target_dir):
    for imagepath in imagepathlist:
        filename = os.path.basename(imagepath)
        shutil.copy(imagepath, os.path.join(target_dir, filename))

if __name__ == "__main__":
    gather_example("../config/config.json", "../video/dami.mp4", "../image")

import face_recognition as fr
import cv2
import numpy as np
from face_verification.face_verification import FaceVerification

def main():
    fv = FaceVerification("config.json")
    known_face_encodings, known_face_names, face_locations, face_encodings, face_names, process_this_frame = fv.face_information()
    video_capture = cv2.VideoCapture(0)

    while True:
        ret, frame = video_capture.read()
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
                                           face_encoding)  # Compare a list of face encodings against a candidate encoding to see if they match.
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

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
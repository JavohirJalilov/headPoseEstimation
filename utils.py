import numpy as np
import cv2
import mediapipe as mp


def required_landmarks(landmarks, w, h, face_2d, face_3d):
    # 1 nose, 33, left eye, 263 right eye, 199 chin, 61 left mouth, 291 right mouth
    ids = [1, 33, 263, 61, 291, 199]
    # print(w, h)
    for idx, lm in enumerate(landmarks):
        x = int(lm.x * w)
        y = int(lm.y * h)
        if idx in ids:
            if idx == 1:
                nose_2d = (lm.x*w, lm.y*h)
                nose_3d = (lm.x*w, lm.y*h, lm.z * 3000)

            # get 2d and 3d coordinates
            face_2d.append([x, y])
            face_3d.append([x, y, lm.z])
    
    # convert to numpy array
    face_2d = np.array(face_2d, dtype=np.float64)
    face_3d = np.array(face_3d, dtype=np.float64)

    return face_2d, face_3d, nose_2d, nose_3d
        
        
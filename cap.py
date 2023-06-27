import mediapipe as mp
import cv2
import numpy as np
from utils import required_landmarks
# Initialize the face mesh solution.
face_mesh = mp.solutions.face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

drawing_spec = mp.solutions.drawing_utils.DrawingSpec(thickness=1, circle_radius=1, color=(0, 255, 0))

# Create a video capture object.
cap = cv2.VideoCapture(0)

while True:
    # Capture the next frame.
    ret, frame = cap.read()
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    # Process the frame with the face mesh solution.
    results = face_mesh.process(frame)

    image.flags.writeable = True
    img_height, img_width , _ = image.shape

    face_2d = []
    face_3d = []
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            face_2d, face_3d, nose_2d, nose_3d = required_landmarks(face_landmarks.landmark, img_width, img_height, face_2d, face_3d)
            x, y = tuple(nose_2d)
            # draw circle on nose
            for x_, y_ in face_2d:
                cv2.circle(image, (int(x_), int(y_)), 5, (0, 255, 0), -1)
            facal_length = 1*img_width
            cam_matrix = np.array([[facal_length, 0, img_width/2],
                                      [0, facal_length, img_height/2],
                                      [0, 0, 1]], dtype=np.float32)
            dist_coeffs = np.zeros((4,1), dtype=np.float32)

            # solve pnp
            success, rotation_vector, translation_vector = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_coeffs, flags=cv2.SOLVEPNP_UPNP)
            
            # get rotation matrix
            rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
            # get angles RQdecomposition
            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rotation_matrix)
            x, y, z = (angles[0]*360, angles[1]*360, angles[2]*360)
            
            # see where the user's head is tiling
            txt = "Look camera"
            if y < -4:
                txt = "Look left"
            elif y > 4:
                txt = "Look right"
            elif x < -4:
                txt = "Look down"
            elif x > 4:
                txt = "Look up"

            # display the nose coordinates
            nose2d_projection, jac = cv2.projectPoints(nose_3d, rotation_vector, translation_vector, cam_matrix, dist_coeffs)

            # find angle between nose line and x axis
            x_angle = nose2d_projection[0][0][0] - nose_2d[0]
            y_angle = nose2d_projection[0][0][1] - nose_2d[1]
            angle = np.arctan2(y_angle, x_angle)
            angle = np.degrees(angle)

            pt1 = (int(nose_2d[0]), int(nose_3d[1]))
            pt2 = (int(nose_2d[0] + y*10), int(nose_2d[1] - x*10))

            cv2.line(image, pt1, pt2, (255, 0, 0), 2)
            cv2.putText(image, txt, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imshow('Frame', image)

    # Check if the user wants to quit.
    key = cv2.waitKey(1)
    if key == 27:
        break

# AUTHOR: ROHIT DAS

import cv2
import mediapipe

# CONSTANTS

# Landmarks from mesh_map.jpg
LEFT_EYE = [353]
RIGHT_EYE = [124]
MIDDLE_FOREHEAD_POINT = [168]


mediapipe_face_mesh = mediapipe.solutions.face_mesh


def detect_eyes(frame):
    face_mesh = mediapipe_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.6, min_tracking_confidence=0.7)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = face_mesh.process(rgb_frame)
    landmark_points = output.multi_face_landmarks
    if landmark_points:
        landmarks = landmark_points[0].landmark
        for i in LEFT_EYE:
            cv2.circle(frame, (int(landmarks[i].x * frame.shape[1]), int(landmarks[i].y * frame.shape[0])), 1,
                       (0, 0, 255), 1)

        for i in RIGHT_EYE:
            cv2.circle(frame, (int(landmarks[i].x * frame.shape[1]), int(landmarks[i].y * frame.shape[0])), 1,
                       (0, 0, 255), 1)

        for i in MIDDLE_FOREHEAD_POINT:
            cv2.circle(frame, (int(landmarks[i].x * frame.shape[1]), int(landmarks[i].y * frame.shape[0])), 1,
                       (0, 0, 255), 1)

    return frame


# Test code for view_landmarks.py

# video_capture = cv2.VideoCapture(0)
#
# while True:
#     _, frame = video_capture.read()
#     frame = cv2.flip(frame, 1)
#     frame = detect_eyes(frame)
#     cv2.imshow('frame', frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# video_capture.release()
# cv2.destroyAllWindows()

import cv2
import mediapipe as mp
import numpy as np
import trimesh


class GlassesRenderer:
    def __init__(self, glasses_model_path):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise Exception("Could not open video device")

        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1,
                                                    min_detection_confidence=0.5)

        self.glasses_model = trimesh.load(glasses_model_path)

    def detect_face_landmarks(self, image):
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_image)
        if not results.multi_face_landmarks:
            return None
        return results.multi_face_landmarks[0]

    def calculate_pose(self, landmarks, width, height):
        # 3D model points
        model_points = np.array([
            (0.0, 0.0, 0.0),  # Nose tip
            (0.0, -330.0, -65.0),  # Chin
            (-225.0, 170.0, -135.0),  # Left eye left corner
            (225.0, 170.0, -135.0),  # Right eye right corner
            (-150.0, -150.0, -125.0),  # Left Mouth corner
            (150.0, -150.0, -125.0)  # Right mouth corner
        ])

        # 2D image points
        image_points = np.array([
            (landmarks.landmark[4].x * width, landmarks.landmark[4].y * height),  # Nose tip
            (landmarks.landmark[152].x * width, landmarks.landmark[152].y * height),  # Chin
            (landmarks.landmark[33].x * width, landmarks.landmark[33].y * height),  # Left eye left corner
            (landmarks.landmark[263].x * width, landmarks.landmark[263].y * height),  # Right eye right corner
            (landmarks.landmark[61].x * width, landmarks.landmark[61].y * height),  # Left Mouth corner
            (landmarks.landmark[291].x * width, landmarks.landmark[291].y * height)  # Right mouth corner
        ], dtype="double")

        # Camera internals
        focal_length = width
        center = (width / 2, height / 2)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]],
             [0, focal_length, center[1]],
             [0, 0, 1]], dtype="double"
        )

        dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
        (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix,
                                                                      dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

        return rotation_vector, translation_vector, camera_matrix, dist_coeffs

    def render_glasses(self, image, landmarks):
        height, width = image.shape[:2]
        (rotation_vector, translation_vector, camera_matrix, dist_coeffs) = self.calculate_pose(landmarks, width,
                                                                                                height)

        glasses_3d = self.glasses_model.vertices
        (glasses_2d, _) = cv2.projectPoints(glasses_3d, rotation_vector, translation_vector, camera_matrix, dist_coeffs)

        left_eye = np.array([landmarks.landmark[33].x, landmarks.landmark[33].y]) * [width, height]
        right_eye = np.array([landmarks.landmark[263].x, landmarks.landmark[263].y]) * [width, height]
        eye_distance = np.linalg.norm(right_eye - left_eye)

        glasses_2d = glasses_2d.reshape(-1, 2)

        # Normalize the glasses coordinates
        glasses_min = glasses_2d.min(axis=0)
        glasses_max = glasses_2d.max(axis=0)
        glasses_center = (glasses_min + glasses_max) / 2
        glasses_size = glasses_max - glasses_min

        # Scale the glasses to match the eye distance
        scale_factor = eye_distance / glasses_size[0]
        glasses_2d = (glasses_2d - glasses_center) * (scale_factor)

        # Position the glasses on the face
        face_center = (left_eye + right_eye) / 2
        glasses_2d += face_center

        result_image = image.copy()

        for face in self.glasses_model.faces:
            points = glasses_2d[face].astype(np.int32)
            cv2.polylines(result_image, [points], isClosed=True, color=(255, 255, 255), thickness=2)

        return result_image

    def run(self):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            landmarks = self.detect_face_landmarks(frame)
            if landmarks:
                frame = self.render_glasses(frame, landmarks)

            cv2.imshow('Glasses Renderer', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()


# Usage
renderer = GlassesRenderer("glassFrame/glasses2.obj")
renderer.run()

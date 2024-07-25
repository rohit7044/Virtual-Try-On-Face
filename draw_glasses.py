import cv2
import mediapipe as mp
import numpy as np
import trimesh


class GlassesRenderer:
    def __init__(self, glasses_model_path):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1,
                                                    min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.glasses_model = trimesh.load(glasses_model_path)

    def calculate_pose(self, landmarks, image_shape):
        height, width = image_shape[:2]
        model_points = np.array([
            (0.0, 0.0, 0.0),  # Nose tip
            (0.0, -330.0, -65.0),  # Chin
            (-225.0, 170.0, -135.0),  # Left eye left corner
            (225.0, 170.0, -135.0),  # Right eye right corner
            (-150.0, -150.0, -125.0),  # Left Mouth corner
            (150.0, -150.0, -125.0)  # Right mouth corner
        ])

        image_points = np.array([
            (landmarks.landmark[4].x * width, landmarks.landmark[4].y * height),
            (landmarks.landmark[152].x * width, landmarks.landmark[152].y * height),
            (landmarks.landmark[33].x * width, landmarks.landmark[33].y * height),
            (landmarks.landmark[263].x * width, landmarks.landmark[263].y * height),
            (landmarks.landmark[61].x * width, landmarks.landmark[61].y * height),
            (landmarks.landmark[291].x * width, landmarks.landmark[291].y * height)
        ], dtype="double")

        focal_length = width
        center = (width / 2, height / 2)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]],
             [0, focal_length, center[1]],
             [0, 0, 1]], dtype="double"
        )

        dist_coeffs = np.zeros((4, 1))
        (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix,
                                                                      dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

        return (rotation_vector, translation_vector, camera_matrix, dist_coeffs)

    def render_glasses(self, image, landmarks):
        height, width = image.shape[:2]
        (rotation_vector, translation_vector, camera_matrix, dist_coeffs) = self.calculate_pose(landmarks, image.shape)

        glasses_3d = self.glasses_model.vertices
        (glasses_2d, _) = cv2.projectPoints(glasses_3d, rotation_vector, translation_vector, camera_matrix, dist_coeffs)

        left_eye = np.array([landmarks.landmark[33].x, landmarks.landmark[33].y]) * [width, height]
        right_eye = np.array([landmarks.landmark[263].x, landmarks.landmark[263].y]) * [width, height]
        eye_distance = np.linalg.norm(right_eye - left_eye)

        glasses_2d = glasses_2d.reshape(-1, 2)

        glasses_min = glasses_2d.min(axis=0)
        glasses_max = glasses_2d.max(axis=0)
        glasses_center = (glasses_min + glasses_max) / 2
        glasses_size = glasses_max - glasses_min

        scale_factor = eye_distance / glasses_size[0] * 1.5  # Slightly larger than eye distance
        glasses_2d = (glasses_2d - glasses_center) * scale_factor

        face_center = (left_eye + right_eye) / 2
        nose_bridge = np.array([landmarks.landmark[168].x, landmarks.landmark[168].y]) * [width, height]
        vertical_offset = (nose_bridge[1] - face_center[1]) * 0.1  # Slight vertical adjustment
        glasses_2d += face_center + [0, vertical_offset]

        # Create a mask for the glasses
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        for face in self.glasses_model.faces:
            points = glasses_2d[face].astype(np.int32)
            cv2.fillConvexPoly(mask, points, 255)

        # Create a semi-transparent overlay for the glasses
        overlay = np.zeros_like(image)
        glasses_color = (255, 255, 255)  # color for glasses
        cv2.fillPoly(overlay, [glasses_2d.astype(np.int32)], glasses_color)

        # Blend the overlay with the original image
        alpha = 1.0  # Adjust this value to change the transparency (0.0 to 1.0)
        result = cv2.addWeighted(image, 1, overlay, alpha, 0)

        # Apply the mask to keep only the glasses area
        mask = cv2.merge([mask, mask, mask])
        result = np.where(mask > 0, result, image)

        return result

    def process_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            frame = self.render_glasses(frame, landmarks)

        return frame


def main():
    glasses_model_path = "glassFrame/glasses2.obj"  # Update this path
    renderer = GlassesRenderer(glasses_model_path)

    cap = cv2.VideoCapture(0)  # Use 0 for default camera

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        frame = renderer.process_frame(frame)

        cv2.imshow('Glasses AR', frame)

        if cv2.waitKey(5) & 0xFF == 27:  # Press 'ESC' to exit
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
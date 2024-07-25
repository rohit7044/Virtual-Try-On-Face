import cv2
import mediapipe as mp
import numpy as np
import trimesh


class GlassesRenderer:
    def __init__(self, glasses_model_path: str) -> None:
        """
        Initialize the GlassesRenderer class.

        Args:
            glasses_model_path (str): The path to the glasses model file.

        Returns:
            None
        """
        # Initialize the MediaPipe face mesh module
        self.mp_face_mesh = mp.solutions.face_mesh

        # Create a face mesh object with specified parameters
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,  # Process real-time video frames
            max_num_faces=1,  # Detect only the closest face
            min_detection_confidence=0.5,  # Minimum confidence for face detection
            min_tracking_confidence=0.5  # Minimum confidence for face tracking
        )

        # Load the glasses model from the given file path
        self.glasses_model = trimesh.load(glasses_model_path)

    def calculate_pose(self, landmarks, image_shape):
        """
        Calculate the pose (rotation and translation vectors) of the face in the image.

        Args:
            landmarks (mediapipe.face_mesh.FaceMesh): The detected landmarks of the face.
            image_shape (tuple): The shape of the image.

        Returns:
            tuple: A tuple containing the rotation vector, translation vector, camera matrix, and distortion coefficients.
        """
        # Extract the height and width of the image
        height, width = image_shape[:2]

        # Define the 3D model points
        model_points = np.array([
            (0.0, 0.0, 0.0),  # Nose tip
            (0.0, -330.0, -65.0),  # Chin
            (-225.0, 170.0, -135.0),  # Left eye left corner
            (225.0, 170.0, -135.0),  # Right eye right corner
            (-150.0, -150.0, -125.0),  # Left Mouth corner
            (150.0, -150.0, -125.0)  # Rightmouth corner
        ], dtype="double")

        # Define the corresponding image points
        image_points = np.array([
            (landmarks.landmark[4].x * width, landmarks.landmark[4].y * height),
            (landmarks.landmark[152].x * width, landmarks.landmark[152].y * height),
            (landmarks.landmark[33].x * width, landmarks.landmark[33].y * height),
            (landmarks.landmark[263].x * width, landmarks.landmark[263].y * height),
            (landmarks.landmark[61].x * width, landmarks.landmark[61].y * height),
            (landmarks.landmark[291].x * width, landmarks.landmark[291].y * height)
        ], dtype="double")

        # Define the camera matrix
        camera_matrix = np.array([
            [width, 0, width / 2],
            [0, width, height / 2],
            [0, 0, 1]
        ], dtype="double")

        # Initialize the distortion coefficients
        dist_coeffs = np.zeros((4, 1))

        # Solve the PnP problem to obtain the rotation and translation vectors
        (success, rotation_vector, translation_vector) = cv2.solvePnP(
            model_points, image_points, camera_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        return rotation_vector, translation_vector, camera_matrix, dist_coeffs

    def render_glasses(self, image, landmarks):
        """
        Render glasses on an image based on the detected landmarks.

        Args:
            image (numpy.ndarray): The input image.
            landmarks (mediapipe.face_mesh.FaceMesh): The detected landmarks of the face.

        Returns:
            numpy.ndarray: The image with the glasses rendered on it.
        """
        # Extract the height and width of the image
        height, width = image.shape[:2]

        # Calculate the pose (rotation and translation vectors) of the face in the image
        rotation_vector, translation_vector, camera_matrix, dist_coeffs = self.calculate_pose(landmarks, image.shape)

        # Project the 3D glasses model onto the image plane
        glasses_3d = self.glasses_model.vertices
        glasses_2d, _ = cv2.projectPoints(glasses_3d, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
        glasses_2d = glasses_2d.reshape(-1, 2)

        # Calculate the position of the left and right eyes
        left_eye = np.array([landmarks.landmark[33].x, landmarks.landmark[33].y]) * [width, height]
        right_eye = np.array([landmarks.landmark[263].x, landmarks.landmark[263].y]) * [width, height]
        eye_distance = np.linalg.norm(right_eye - left_eye)

        # Calculate the center and size of the glasses
        glasses_min = glasses_2d.min(axis=0)
        glasses_max = glasses_2d.max(axis=0)
        glasses_center = (glasses_min + glasses_max) / 2
        glasses_size = glasses_max - glasses_min

        # Scale the glasses to fit the eye distance
        scale_factor = eye_distance / glasses_size[0] * 1.5
        glasses_2d = (glasses_2d - glasses_center) * scale_factor

        # Calculate the center of the face
        face_center = (left_eye + right_eye) / 2

        # Calculate the vertical offset of the glasses based on the nose bridge position
        nose_bridge = np.array([landmarks.landmark[168].x, landmarks.landmark[168].y]) * [width, height]
        vertical_offset = (nose_bridge[1] - face_center[1]) * 0.1
        glasses_2d += face_center + [0, vertical_offset]

        # Create a mask for the glasses
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        for face in self.glasses_model.faces:
            points = glasses_2d[face].astype(np.int32)
            cv2.fillConvexPoly(mask, points, 255)

        # Create an overlay with the glasses color
        overlay = np.zeros_like(image)
        glasses_color = (255, 255, 255)
        cv2.fillPoly(overlay, [glasses_2d.astype(np.int32)], glasses_color)

        # Combine the image and overlay with the glasses
        alpha = 1.0
        result = cv2.addWeighted(image, 1, overlay, alpha, 0)

        # Apply the mask to the result
        mask = cv2.merge([mask, mask, mask])
        result = np.where(mask > 0, result, image)

        return result

    def process_frame(self, frame):
        """
        Process a frame from a video stream to render glasses on the detected face.

        Args:
            frame (numpy.ndarray): The input frame to process.

        Returns:
            numpy.ndarray: The processed frame with glasses rendered on the detected face.
        """
        # Convert the frame to RGB color space
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame using the face mesh model
        results = self.face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            # Get the landmarks of the first detected face
            landmarks = results.multi_face_landmarks[0]

            # Render glasses on the detected face
            frame = self.render_glasses(frame, landmarks)

        # Return the processed frame
        return frame


def main():
    """
    This function sets up a video capture, renders glasses on the face in the video,
    and displays the result in a window. The function runs until the user closes the window.
    """
    # Path to the glasses model file
    glasses_model_path = "glassFrame/oculos.obj"

    # Create a GlassesRenderer object with the glasses model
    renderer = GlassesRenderer(glasses_model_path)

    # Open the default camera
    cap = cv2.VideoCapture(0)

    # Process each frame of the video
    while cap.isOpened():
        # Read the next frame from the video
        success, frame = cap.read()

        # If the frame is empty, ignore it
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Render the glasses on the face in the frame
        frame = renderer.process_frame(frame)

        # Display the resulting frame in a window
        cv2.imshow('Glasses AR', frame)

        # If the 'Esc' key is pressed, stop the loop
        if cv2.waitKey(5) & 0xFF == 27:
            break

    # Release the video capture and close all windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

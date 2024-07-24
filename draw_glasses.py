import cv2
import mediapipe as mp
import numpy as np
import trimesh
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

# Initialize webcam
cap = cv2.VideoCapture(0)

# Load 3D glasses model
glasses_model = trimesh.load('glassFrame/glasses2.obj')

# Define key points (adjust these indices based on MediaPipe Face Mesh layout)
KEY_POINTS = {
    'mid_eye': 168,
    'left_eye': 143,
    'nose_bottom': 2,
    'right_eye': 372
}
# Adjustable parameters
SCALE_FACTOR = 10  # Adjust this to change the size of the glasses
VERTICAL_OFFSET = 2  # Adjust this to move the glasses up/down
DEPTH_OFFSET = 2 # Adjust this to move the glasses closer to/further from the face


# OpenGL setup
def init_gl(width, height):
    glClearColor(0.0, 0.0, 0.0, 0.0)
    glClearDepth(1.0)
    glDepthFunc(GL_LESS)
    glEnable(GL_DEPTH_TEST)
    glShadeModel(GL_SMOOTH)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45.0, float(width) / float(height), 0.1, 100.0)
    glMatrixMode(GL_MODELVIEW)


def draw_glasses(position, rotation, scale):
    glPushMatrix()
    glTranslatef(position[0], position[1], position[2])
    glRotatef(rotation[0], 1, 0, 0)
    glRotatef(rotation[1], 0, 1, 0)
    glRotatef(rotation[2], 0, 0, 1)
    glScalef(scale, scale, scale)

    glBegin(GL_TRIANGLES)
    for face in glasses_model.faces:
        for vertex_index in face:
            vertex = glasses_model.vertices[vertex_index]
            glVertex3f(vertex[0], vertex[1], vertex[2])
    glEnd()

    glPopMatrix()


def process_frame(frame: np.ndarray) -> np.ndarray:
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0].landmark

        # Get key points
        mid_eye = np.array([
            face_landmarks[KEY_POINTS['mid_eye']].x,
            -face_landmarks[KEY_POINTS['mid_eye']].y,
            -face_landmarks[KEY_POINTS['mid_eye']].z
        ])
        left_eye = np.array([
            face_landmarks[KEY_POINTS['left_eye']].x,
            -face_landmarks[KEY_POINTS['left_eye']].y,
            -face_landmarks[KEY_POINTS['left_eye']].z
        ])
        right_eye = np.array([
            face_landmarks[KEY_POINTS['right_eye']].x,
            -face_landmarks[KEY_POINTS['right_eye']].y,
            -face_landmarks[KEY_POINTS['right_eye']].z
        ])
        nose_bottom = np.array([
            face_landmarks[KEY_POINTS['nose_bottom']].x,
            -face_landmarks[KEY_POINTS['nose_bottom']].y,
            -face_landmarks[KEY_POINTS['nose_bottom']].z
        ])

        # Calculate position, rotation, and scale
        position = mid_eye + np.array([0, VERTICAL_OFFSET, DEPTH_OFFSET])
        up_vector = mid_eye - nose_bottom
        up_vector /= np.linalg.norm(up_vector)

        forward_vector = np.cross(right_eye - left_eye, up_vector)
        forward_vector /= np.linalg.norm(forward_vector)

        right_vector = np.cross(up_vector, forward_vector)

        rotation_matrix = np.column_stack((right_vector, up_vector, forward_vector))
        rotation = trimesh.transformations.euler_from_matrix(rotation_matrix)

        scale = np.linalg.norm(right_eye - left_eye) * SCALE_FACTOR

        # Clear the frame and depth buffer
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Set up the camera
        glLoadIdentity()
        gluLookAt(0, 0, 0, 0, 0, -1, 0, 1, 0)

        # Draw the glasses
        draw_glasses(position, np.degrees(rotation), scale)

        # Read the OpenGL buffer
        glPixelStorei(GL_PACK_ALIGNMENT, 1)
        data = glReadPixels(0, 0, frame.shape[1], frame.shape[0], GL_RGBA, GL_UNSIGNED_BYTE)
        image = np.frombuffer(data, dtype=np.uint8).reshape(frame.shape[0], frame.shape[1], 4)
        image = np.flipud(image)

        # cv2.imshow('Face Mesh with 3D Glasses', image)
        # cv2.waitKey(0)

        # Blend the OpenGL render with the original frame
        mask = image[:, :, 3] / 255.0
        for c in range(3):
            frame[:, :, c] = frame[:, :, c] * (1 - mask) + image[:, :, c] * mask

    return frame


def main():
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Initialize OpenGL
    glutInit()
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
    glutInitWindowSize(width, height)
    glutCreateWindow(b"Face Mesh with 3D Glasses")
    init_gl(width, height)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        frame = process_frame(frame)

        cv2.imshow('Face Mesh with 3D Glasses', frame)

        if cv2.waitKey(5) & 0xFF == 27:  # Press ESC to exit
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

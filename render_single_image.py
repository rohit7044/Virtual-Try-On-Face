import cv2
import mediapipe as mp
import numpy as np
import trimesh
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *


class GlassesRenderer:
    def __init__(self, image_path, glasses_model_path):
        self.image = cv2.imread(image_path)
        self.height, self.width = self.image.shape[:2]

        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1,
                                                    min_detection_confidence=0.5)

        self.glasses_model = trimesh.load(glasses_model_path)

    def detect_face_landmarks(self):
        rgb_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_image)
        if not results.multi_face_landmarks:
            raise Exception("No face detected in the image")
        return results.multi_face_landmarks[0]

    def setup_opengl(self):
        glutInit()
        glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
        glutInitWindowSize(self.width, self.height)
        glutCreateWindow(b"Glasses Render")

        glEnable(GL_DEPTH_TEST)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, (self.width / self.height), 0.1, 50.0)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

    def draw_landmarks(self, landmarks):
        landmark_image = self.image.copy()
        h, w = landmark_image.shape[:2]
        for idx, lm in enumerate(landmarks.landmark):
            x, y = int(lm.x * w), int(lm.y * h)
            cv2.circle(landmark_image, (x, y), 1, (0, 255, 0), -1)
            cv2.putText(landmark_image, str(idx), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)

        cv2.imshow("Landmarks", landmark_image)
        cv2.waitKey(0)

    def render_glasses(self, landmarks):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()

        # Position the camera
        glTranslatef(0, 0, -5)

        # Get key points for glasses positioning and scaling
        nose_bridge = np.array([landmarks.landmark[168].x, landmarks.landmark[168].y, landmarks.landmark[168].z])
        left_eye = np.array([landmarks.landmark[33].x, landmarks.landmark[33].y, landmarks.landmark[33].z])
        right_eye = np.array([landmarks.landmark[263].x, landmarks.landmark[263].y, landmarks.landmark[263].z])
        left_temple = np.array([landmarks.landmark[234].x, landmarks.landmark[234].y, landmarks.landmark[234].z])
        right_temple = np.array([landmarks.landmark[454].x, landmarks.landmark[454].y, landmarks.landmark[454].z])

        # Calculate glasses position
        glasses_center = (left_eye + right_eye) / 2

        # Calculate scale based on face width
        face_width = np.linalg.norm(right_temple - left_temple)

        # Calculate rotation based on eye positions
        eye_vector = right_eye - left_eye
        angle = np.arctan2(eye_vector[1], eye_vector[0])

        # Adjust these values to fine-tune the fit
        scale_multiplier = 1.1  # Adjust this to make glasses slightly larger or smaller
        y_offset = 0.02  # Adjust this to move glasses up or down
        z_offset = 0.05  # Adjust this to move glasses forward or backward

        glPushMatrix()

        # Position the glasses
        glTranslatef(glasses_center[0] - 0.5, -glasses_center[1] + 0.5 + y_offset, -glasses_center[2] + z_offset)

        # Rotate the glasses to match face orientation
        glRotatef(np.degrees(angle), 0, 0, 1)

        # Scale the glasses
        glScalef(face_width * scale_multiplier, face_width * scale_multiplier, face_width * scale_multiplier)

        # Center the glasses model if needed
        glTranslatef(-0.5, 0, 0)  # Uncomment and adjust if your model isn't centered

        # Render the glasses model
        if isinstance(self.glasses_model, trimesh.Scene):
            for geometry in self.glasses_model.geometry.values():
                glBegin(GL_TRIANGLES)
                for face in geometry.faces:
                    for vertex_index in face:
                        vertex = geometry.vertices[vertex_index]
                        glVertex3f(*vertex)
                glEnd()
        else:
            glBegin(GL_TRIANGLES)
            for face in self.glasses_model.faces:
                for vertex_index in face:
                    vertex = self.glasses_model.vertices[vertex_index]
                    glVertex3f(*vertex)
            glEnd()

        glPopMatrix()
        glutSwapBuffers()

    def save_result(self):
        glReadBuffer(GL_FRONT)
        pixels = glReadPixels(0, 0, self.width, self.height, GL_RGBA, GL_UNSIGNED_BYTE)
        image = np.frombuffer(pixels, dtype=np.uint8).reshape(self.height, self.width, 4)
        image = cv2.flip(image, 0)
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)

        # Blend the rendered glasses with the original image
        mask = image[:, :, 0] > 0
        self.image[mask] = image[mask]

        # cv2.imwrite("result.jpg", self.image)
        cv2.imshow("Result", self.image)
        cv2.waitKey(0)

    def run(self):
        landmarks = self.detect_face_landmarks()
        self.setup_opengl()
        # self.draw_landmarks(landmarks)
        self.render_glasses(landmarks)
        self.save_result()


# Usage
renderer = GlassesRenderer("WIN_20240724_19_43_05_Pro.jpg", "glassFrame/glasses2.obj")
renderer.run()

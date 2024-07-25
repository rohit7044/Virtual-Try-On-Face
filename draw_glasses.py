import cv2
import mediapipe as mp
import numpy as np
import trimesh
import OpenGL.GL as gl
import OpenGL.GLUT as glut
import OpenGL.GLU as glu
import pyrr


class VirtualGlasses:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.cap = cv2.VideoCapture(0)
        success, frame = self.cap.read()
        if not success:
            raise Exception("Could not initialize camera")

        self.frame_height, self.frame_width = frame.shape[:2]

        self.glasses_model = None
        self.load_glasses_model()

        # Initialize GLUT
        glut.glutInit()
        glut.glutInitDisplayMode(glut.GLUT_DOUBLE | glut.GLUT_RGB | glut.GLUT_DEPTH)
        glut.glutInitWindowSize(self.frame_width, self.frame_height)
        glut.glutCreateWindow(b"Virtual Glasses")

        # Set up OpenGL context
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        glu.gluPerspective(45, (self.frame_width / self.frame_height), 0.1, 50.0)
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glLoadIdentity()

    def load_glasses_model(self):
        # Load your 3D glasses model here
        # For example:
        self.glasses_model = trimesh.load('glassFrame/glasses2.obj')
        pass

    def process_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                self.render_glasses(frame, face_landmarks)

        return frame

    def render_glasses(self, frame, face_landmarks):
        gl.glPushMatrix()

        # 1. Extract key facial landmarks
        mid_eye = np.array(
            [face_landmarks.landmark[168].x, face_landmarks.landmark[168].y, face_landmarks.landmark[168].z])
        left_eye = np.array(
            [face_landmarks.landmark[143].x, face_landmarks.landmark[143].y, face_landmarks.landmark[143].z])
        nose_bottom = np.array(
            [face_landmarks.landmark[2].x, face_landmarks.landmark[2].y, face_landmarks.landmark[2].z])
        right_eye = np.array(
            [face_landmarks.landmark[372].x, face_landmarks.landmark[372].y, face_landmarks.landmark[372].z])

        # 2. Calculate glasses position
        glasses_position = mid_eye

        # 3. Calculate glasses orientation
        forward = np.array([0, 0, 1])  # Assuming the glasses model faces forward along positive Z
        up = mid_eye - nose_bottom
        up = up / np.linalg.norm(up)  # Normalize the up vector
        right = np.cross(up, forward)
        right = right / np.linalg.norm(right)  # Normalize the right vector
        forward = np.cross(right, up)  # Recalculate forward to ensure orthogonality

        # Create rotation matrix
        rotation_matrix = np.column_stack((right, up, forward))

        # 4. Calculate glasses scale
        eye_distance = np.linalg.norm(left_eye - right_eye)
        glasses_scale = eye_distance * 1.5  # Adjust this factor as needed

        # 5. Set up OpenGL matrices
        gl_position = (glasses_position[0] - 0.5, -glasses_position[1] + 0.5, -glasses_position[2])

        # Create model matrix
        # Use the same scale for all dimensions
        model_matrix = pyrr.matrix44.create_from_scale([glasses_scale, glasses_scale, glasses_scale])
        model_matrix = pyrr.matrix44.multiply(model_matrix, pyrr.matrix44.create_from_matrix33(rotation_matrix))
        model_matrix = pyrr.matrix44.multiply(model_matrix, pyrr.matrix44.create_from_translation(gl_position))

        # Apply the model matrix
        gl.glMultMatrixf(model_matrix)

        # 6. Render the glasses model
        if self.glasses_model:
            for mesh in self.glasses_model.geometry:
                gl.glBegin(gl.GL_TRIANGLES)
                for face in mesh.faces:
                    for vertex_index in face:
                        vertex = mesh.vertices[vertex_index]
                        gl.glVertex3f(*vertex)
                gl.glEnd()

        gl.glPopMatrix()

    def display(self):
        success, frame = self.cap.read()
        if success:
            frame = self.process_frame(frame)

            # Convert frame to OpenGL texture
            gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
            gl.glEnable(gl.GL_TEXTURE_2D)
            texture_id = gl.glGenTextures(1)
            gl.glBindTexture(gl.GL_TEXTURE_2D, texture_id)
            gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB, frame.shape[1], frame.shape[0], 0, gl.GL_BGR,
                            gl.GL_UNSIGNED_BYTE, frame)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)

            # Draw the texture on a quad
            gl.glBegin(gl.GL_QUADS)
            gl.glTexCoord2f(0, 0);
            gl.glVertex2f(-1, -1)
            gl.glTexCoord2f(1, 0);
            gl.glVertex2f(1, -1)
            gl.glTexCoord2f(1, 1);
            gl.glVertex2f(1, 1)
            gl.glTexCoord2f(0, 1);
            gl.glVertex2f(-1, 1)
            gl.glEnd()

            gl.glDisable(gl.GL_TEXTURE_2D)
            gl.glDeleteTextures([texture_id])

        glut.glutSwapBuffers()

    def idle(self):
        glut.glutPostRedisplay()

    def run(self):
        glut.glutDisplayFunc(self.display)
        glut.glutIdleFunc(self.idle)
        glut.glutMainLoop()

    def __del__(self):
        self.cap.release()


if __name__ == "__main__":
    app = VirtualGlasses()
    app.run()

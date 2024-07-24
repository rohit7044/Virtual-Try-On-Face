import numpy as np
import trimesh
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

# Load a 3D glasses model using trimesh
glasses_model = trimesh.load('glassFrame/glasses2.obj')

# Simulated landmark points (for example purposes)
# These points represent the positions of the key facial landmarks
# in 3D space. Adjust these points to simulate different facial positions.
landmarks = {
    'mid_eye': np.array([0, 0, 0]),
    'left_eye': np.array([-0.5, 0, 0]),
    'right_eye': np.array([0.5, 0, 0]),
    'nose_bottom': np.array([0, -0.5, 0])
}

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

def calculate_transformations(landmarks):
    mid_eye = landmarks['mid_eye']
    left_eye = landmarks['left_eye']
    right_eye = landmarks['right_eye']
    nose_bottom = landmarks['nose_bottom']

    # Calculate position, rotation, and scale
    position = mid_eye
    up_vector = mid_eye - nose_bottom
    up_vector /= np.linalg.norm(up_vector)

    forward_vector = np.cross(right_eye - left_eye, up_vector)
    forward_vector /= np.linalg.norm(forward_vector)

    right_vector = np.cross(up_vector, forward_vector)

    rotation_matrix = np.column_stack((right_vector, up_vector, forward_vector))
    rotation = trimesh.transformations.euler_from_matrix(rotation_matrix)

    scale = np.linalg.norm(right_eye - left_eye)

    return position, rotation, scale

def display():
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()
    gluLookAt(0, 0, 5, 0, 0, 0, 0, 1, 0)

    position, rotation, scale = calculate_transformations(landmarks)

    print("Position:", position)
    print("Rotation (degrees):", np.degrees(rotation))
    print("Scale:", scale)

    draw_glasses(position, np.degrees(rotation), scale)

    glutSwapBuffers()

def main():
    width, height = 800, 600
    glutInit()
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
    glutInitWindowSize(width, height)
    glutCreateWindow(b"3D Glasses with Transformations")
    init_gl(width, height)
    glutDisplayFunc(display)
    glutIdleFunc(display)
    glutMainLoop()

if __name__ == "__main__":
    main()

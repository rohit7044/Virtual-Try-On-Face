# 3D Glasses Try-On with Live Camera Feed

## Overview

This repository demonstrates a real-time 3D glasses try-on application using a live camera feed. The system integrates MediaPipe for facial landmark detection and Trimesh for handling 3D models of glasses. The project consists of scripts for both single image processing and live video processing to render 3D glasses on detected faces.

## Setup

### Creating the Environment


1. **Create the Anaconda Environment:**

    ```bash
    conda create --name virtual_try_on_face python=3.10
    ```

2. **Activate the Environment:**

    ```bash
    conda activate virtual_try_on_face
    ```

3. **Install Dependencies from `requirements.txt`:**

    Make sure you have a `requirements.txt` file in your repository with the necessary packages. Use the following command to install them:

    ```bash
    pip install -r requirements.txt
    ```

### Requirements

Ensure you have the following packages listed in your `requirements.txt`:

- `mediapipe`
- `opencv-python`
- `numpy`
- `trimesh`

## Code Explanation
#### You can use various other 3D glass models. Just change the path in the `GlassesRenderer` class.
### 1. Facial Landmark Detection

**File: `view_landmarks.py`**

This script detects and visualizes facial landmarks using MediaPipe. It identifies key facial points such as the eyes and forehead and marks them on the input image.

### How to Run:

Run the script to visualize the landmarks from your camera:

```python
python view_landmarks.py 
```
### 2. Single Image Glasses Rendering

**File: `render_single_image.py`**

This script renders 3D glasses on a single image based on the detected facial landmarks. It uses Trimesh to load the 3D glasses model and aligns it with the detected facial features.

### How to Run:

Run the script and pass an image as an input to visualize the landmarks:

```python
python render_single_image.py 
```
### 3. Live Camera Glasses Rendering

**File: `draw_glasses.py`**

This script captures live video from a webcam and renders 3D glasses in real-time. It uses MediaPipe to detect facial landmarks and Trimesh for rendering the glasses.

### How to Run:

Execute the script to start the live video feed with glasses rendering:

```python
python draw_glasses.py
```
## Possible Drawbacks
**Frame Alignment**: The current implementation may have issues with precise alignment of the glasses, particularly around the ears. This misalignment can affect the visual accuracy of the rendered glasses.<br>
### Feel free to contribute to the project by addressing these drawbacks or improving the functionality!
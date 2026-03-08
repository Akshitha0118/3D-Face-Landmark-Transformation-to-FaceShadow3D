# 3D-Face-Landmark-Transformation-to-FaceShadow3D

A real-time computer vision project that detects facial landmarks using MediaPipe Face Mesh and applies a transformation matrix to visualize modified 3D facial landmarks.

This project demonstrates how facial landmarks can be manipulated using simple matrix transformations and visualized in real-time using OpenCV.

---

## Features

- Real-time face detection using webcam
- 468 facial landmark detection using MediaPipe Face Mesh
- 3D landmark transformation using a scaling matrix
- Landmark visualization on video frames
- Real-time processing with OpenCV

---

## Technologies Used

- Python
- OpenCV
- MediaPipe
- NumPy

---

## How It Works

1. The webcam captures video frames.
2. MediaPipe Face Mesh detects facial landmarks (468 points).
3. The landmarks are converted into a NumPy array.
4. A transformation matrix is applied to scale the landmarks.
5. The transformed landmarks are drawn onto the frame.

---


# Multi-Face Employee Recognition System
A real-time security and attendance application that identifies multiple people in a single frame. Using DeepFace for facial recognition and Haar Cascades for detection, the system distinguishes between registered employees and unauthorized individuals.

> Features
Multithreaded Processing: Uses Python's threading and Lock to run facial recognition in the background, ensuring a smooth 720p video feed without lag.

Multiple Face Tracking: Detects several faces simultaneously using OpenCV and processes each against the database.

Dynamic Visual Feedback: * Green Box: Identified Employee (Displays Name).

Red Box: Unknown/Unauthorized Person.

Distance-Based Verification: Uses a configurable threshold (0.55) to ensure high-confidence matches before granting "Employee" status.

> Tech Stack
Deep Learning Library: DeepFace (VGG-Face, Facenet, etc.).

Computer Vision: OpenCV.

Optimization: Threading & Concurrent Execution.

> Project Structure
group_recog.py: The core script handling the camera feed, threading, and recognition logic.

employees/: (Required) A folder containing subdirectories or images of authorized personnel.

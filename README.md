This project combines a face recognition system with a graphical user interface (GUI) for a login page. It uses OpenCV for image processing and face recognition, and Tkinter for the GUI.

#Features
Face Recognition: Utilizes the LBPH (Local Binary Patterns Histograms) face recognizer from OpenCV.
Login Interface: A user-friendly login interface built with Tkinter.
Image Display: Displays an image in the login interface.
Real-time Face Detection: Detects faces in real-time using the webcam.
Requirements
Python 3.x
OpenCV
NumPy
PIL (Pillow)
Tkinter


#Install dependencies:

sh
Copy code
pip install opencv-python numpy pillow
Ensure the following files are present:

haarcascade_frontalface_default.xml: Required for face detection.
atm.jpg: The image displayed in the login interface.
A dataset of grayscale face images in the TrainingImage directory.

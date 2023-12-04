# WatchDog-Gamma

## Overview
WatchDog-Gamma is a state-of-the-art face detection and recognition system that uses edge computing to efficiently process and transmit facial data. The project is focused on performing initial computations on a Raspberry Pi, then sending lightweight encoding to a backend server for recognition.

## Edge Computing and Data Transmission : Sensors Module
### Face Detection on Pi:
* **Components Used**: 
  * Raspberry Pi 3 - arm64, clocked at 1200 MHz.
  * USB camera for real-time face capture.

* **Models and Technologies**:
  * **Face Detection**: Utilizes Google's Mediapipe - BlazeFace, which is based on a CNN-SSD architecture.
  * **Face Encodings**: Currently uses OpenFace. However, an update is planned next week to replace it with Face_recognition powered by DNN from OpenCV.
  * **Face Recognition**: Will be conducted using Face Recognition DNN from OpenCV.

### Algorithmic Flow:
#### On Raspberry Pi:
1. Detect faces in a real-time stream via a USB camera.
2. Encode these faces into a 128D tensor.
3. Transmit the tensor values to the backend server for further processing and recognition.

#### On Server/Laptop/Backend:
1. Collected facial data images (80-100 frames per person currently) and converted them into embeddings.
2. These embeddings are stored in a single pickle file, maintaining a database of recognized individuals/patients.
3. The received encoding data from the Pi (typically 700-1000 Bytes, which offers significant bandwidth savings compared to transmitting a whole 1MB image) is then processed.
4. Face recognition is carried out using either:
   * An SVM model for face classification.
   * Libraries from Face Recognition for classification using the method `face_recognition.compare_faces(data["encodings"], encoding)`.

By leveraging this efficient process, we ensure high-speed and accurate face detection and recognition using edge computing!

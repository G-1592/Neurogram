# Neurogram
An interactive 3D MRI tumour visualisation system that combines medical imaging, computer vision and 3D graphics.

## About
The system takes brain MRI data alongside tumour segmentation data and reconstructs them into interactive 3D meshes. A webcam and hand-tracking model then allow the user to interact with the visualisation using hand gestures rather than a mouse. An additional Pepper's Ghost display of the brain mesh was added to enhance the visual experience for the user.

## Demo Videos


## How it works


## Key features
**Calibration system** - Sets user-specific finger bend thresholds using angle and distance data.

**Practice mode** - Checks real-time finger position accuracy for each flute note using hand tracking and hold-to-confirm validation. Optimised for beginner one-octave practice.

**Posture mode** - Tracks embouchure and head alignment using face tracking.

**Gesture-controlled buttons** - Easy touch-free navigation between different modes.


## Screenshots
<table>
  <tr>
    <td align="center">
      
      <br> Practice Mode 
      </td>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/a0100416-095f-4aae-adb5-9ebeea77826c" width="250">
      <br> Calibration System
      </td>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/f2f0c4cd-d0a2-4bf9-b8cc-78b1cd1a22d3" width="250">
      <br> Face Landmark Analysis
      </td>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/96e35cbe-32cc-468f-8712-a039010ca02e" width="250">
      <br> Posture Mode
      </td>
  </tr>
</table>

## Installing the system

1. ### Required Libraries
   - OpenCV
   - MediaPipe
   - NumPy
     
   Install the required libraries using:
   ```bash
   pip install opencv-python mediapipe numpy
   ```

2. ### Required Model Files
   The system also requires the official MediaPipe Hand Landmarker and Face Landmarker model files:
   - hand_landmarker.task - available from the [MediaPipe Hand Landmarker documentation](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/python)
   - face_landmarker.task - available from the [MediaPipe Face Landmarker documentation](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker/python)
   
   Place both files in the main project directory.

3. ### Running the system
   Once the libraries and model files are installed:
   
   **Using an IDE:** Run `main.py`.
   
   **Using the terminal:** 
    ```bash
   python main.py
    ```

## Acknowledgements
- [MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/guide) for the hand and face landmark detection tools and models used in this project.

## Findings
The aim of this project was to see how meaningfully computer vision alone can analyse live flute performance. The results were more accurate than I initially expected! 

However, without audio input, there were several limitations. For example, distinguishing between certain notes can be difficult when the relevant fingering information is not visible to the camera. This was particularly evident with notes C3 and B4 as the thumb behind the flute (which is responsible for changing the pitch) is unseen.

I also found that the right pinky finger was sometimes not detected fully. Although this did not significantly change the recognition of most notes tested, it did reduce the accuracy for detecting D sharp as this is the only note that relies on the right pinky finger to change pitch.

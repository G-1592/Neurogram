# Neurogram
An interactive 3D MRI tumour visualisation system that combines medical imaging, computer vision and 3D graphics.

## About
The system takes brain MRI data alongside tumour segmentation data and reconstructs them into interactive 3D meshes. A webcam and hand-tracking model then allow the user to interact with the visualisation using hand gestures rather than a mouse. An additional Pepper's Ghost display of the brain mesh was added to enhance the visual experience.

## Demo Videos


## How it works


## Key features



## Screenshots
<table>
  <tr>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/a69e45e0-df6a-48fe-80c9-a156bdd15e3a" width="250">
      <br> Brain and Tumour Meshes
      </td>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/2fc4f7ad-e417-42b6-a440-3efef15b3b19" width="250">
      <br> Zoom Control
      </td>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/b5c56897-6a6e-4d20-b57f-e4548a229387" width="250">
      <br> Annotation Mode
      </td>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/ef38a0a8-5550-469a-b74e-196206378f17" width="250">
      <br> Manual Rotation
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

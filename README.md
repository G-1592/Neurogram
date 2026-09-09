# Neurogram
An interactive 3D MRI tumour visualisation system that combines medical imaging, computer vision and 3D graphics.

⚠️🚨**IMPORTANT NOTICE:** *This project is currently undergoing modularisation. At present, all code is contained within the main.py file. Modularisation is expected to be completed very soon.* 🚨⚠️

## About
The system takes brain MRI data alongside tumour segmentation data and reconstructs them into interactive 3D meshes. A webcam and hand-tracking model then allow the user to interact with the visualisation using hand gestures rather than a mouse. An additional Pepper's Ghost display of the brain mesh was added to enhance the visual experience.

## Demo Videos
A very quick demo video (so that it fits in the README!). Longer demo videos are available on my LinkedIn below:

2D Demo video: https://lnkd.in/p/eB-bCJ2u                          3D Pepper's Ghost: https://lnkd.in/p/e8M5qwks

https://github.com/user-attachments/assets/36f6b5c0-deab-4fa0-baca-409dbad74fd2

## How it works
The project first loads a BraTS MRI scan from the file-selection menu, controlled using the "s", "w", Backspace, and Enter keys. It uses the tumour segmentation data (-seg file) and the whole-brain MRI data (-t1c file), which are loaded using NiBabel and converted into two 3D meshes using PyVista: a tumour mesh displayed in red and a brain mesh displayed in blue. This makes it easier to distinguish between the brain and tumour during visualisation.

OpenCV and MediaPipe are then used to track the user's hand and recognise different gestures, with each gesture corresponding to a specific control (see below). From the MRI data and voxel data, brain and tumour volumes are calculated. These calculations include the mesh volume, estimated volume, and percentage difference between the two measurements. The user can press "r" to reset the system and select a new patient MRI.

## Key features
<table>
  <tr>
    <td>
      <img width="500" height="306" alt="Manual" src="https://github.com/user-attachments/assets/755bde11-37dd-429d-96eb-80dfff3cb923" />
    </td>
    <td>
           <strong>Hand Gesture Controls</strong><br>
    Control the 3D brain and tumour models using intuitive hand gestures, including zoom, manual rotation, automatic rotation, and tumour isolation.<br><br>
              <strong>Annotation Mode</strong><br>
    Annotate the brain model using your index finger, with six colour options available.<br><br>
              <strong>Brain & Tumour Calculations</strong><br>
    Calculates the 3D mesh volume, estimates the volume from the original MRI segmentation, and displays the percentage difference between the two measurements.
  
  </td>
  </tr>
</table>

## Screenshots
<table>
  <tr>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/a69e45e0-df6a-48fe-80c9-a156bdd15e3a" width="250" height="180" style="object-fit: cover;">
      <br> Brain + Tumour Meshes
      </td>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/2fc4f7ad-e417-42b6-a440-3efef15b3b19" width="250" height="180" style="object-fit: cover;">
      <br> Zoom Control
      </td>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/b5c56897-6a6e-4d20-b57f-e4548a229387" width="250" height="180" style="object-fit: cover;">
      <br> Annotation Mode
      </td>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/ef38a0a8-5550-469a-b74e-196206378f17" width="250" height="180" style="object-fit: cover;">
      <br> Manual Rotation
      </td>
  </tr>
</table>

## Installing the system

1. ### Required Libraries
   - OpenCV
   - MediaPipe
   - NumPy
   - NiBabel
   - PyVista (and VTK indirectly through PyVista)
   - os
     
   Install the required libraries using:
   ```bash
   pip install opencv-python mediapipe numpy nibabel pyvista os
   ```

2. ### Required Model Files
  The system also requires the official MediaPipe Hand Landmarker files:
   - hand_landmarker.task - available from the [MediaPipe Hand Landmarker documentation](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/python)
   
   Place file in the main project directory.

3. ### Uploading the MRI data
  This project was developed and tested using data from the **BraTS 2023 Adult Glioma (BraTS-GLI) dataset**.

  The MRI data is not included in this repository.

  Users must obtain the BraTS dataset from the official source and place all the files in the directory named "MRI Scans".
  
# - Users must acknowledge and cite the BraTS dataset according to the dataset's citation requirements. 

4. ### Running the system
   Once the libraries and model files are installed:
   
   **Using an IDE:** Run `main.py`.
   
   **Using the terminal:** 
    ```bash
   python main.py
    ```

## Acknowledgements
- [MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/guide) for the hand landmark detection tools and models used in this project.
- This project was developed and tested using data from the **BraTS 2023 Adult Glioma (BraTS-GLI) dataset**. The MRI data is not included in this repository.

## Citations
[1] U.Baid, et al., The RSNA-ASNR-MICCAI BraTS 2021 Benchmark on Brain Tumor Segmentation and Radiogenomic Classification, arXiv:2107.02314, 2021.

[2] B. H. Menze, A. Jakab, S. Bauer, J. Kalpathy-Cramer, K. Farahani, J. Kirby, et al. "The Multimodal Brain Tumor Image Segmentation Benchmark (BRATS)", IEEE Transactions on Medical Imaging 34(10), 1993-2024 (2015) DOI: 10.1109/TMI.2014.2377694

[3] S. Bakas, H. Akbari, A. Sotiras, M. Bilello, M. Rozycki, J.S. Kirby, et al., "Advancing The Cancer Genome Atlas glioma MRI collections with expert segmentation labels and radiomic features", Nature Scientific Data, 4:170117 (2017) DOI: 10.1038/sdata.2017.117

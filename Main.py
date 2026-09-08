
# IMPORTS

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import time
import math
import numpy as np
import nibabel as nib
import pyvista as pv
import os

# HAND LANDMARKER MODEL + DEFINING VARIABLES


# Loading the Hand Landmarker model
base_options = python.BaseOptions(
    model_asset_path='hand_landmarker.task'
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

overlay = cv2.imread("Neurogram.png")
overlay_flip = cv2.flip(overlay, 1)
overlay_flip = cv2.resize(overlay_flip, (640, 120))
manual = cv2.imread("Manual.png")
manual = cv2.resize(manual, (400, 200))

fingers = {
    "Thumb": [1, 2, 3],
    "Index": [5, 6, 7],
    "Middle": [9, 10, 11],
    "Ring": [13, 14, 15],
    "Pinky": [17, 18, 19]
}

# Defining the hand lines between hand landmarks
hand_lines = [
    (0, 1), (1, 2), (2, 3), (3, 4),       # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),       # Index finger
    (5, 9), (9, 10), (10, 11), (11, 12),     # Middle finger
    (9, 13), (13, 14), (14, 15), (15, 16),    # Ring finger
    (13, 17), (17, 18), (18, 19), (19, 20),   # Pinky finger
    (0, 17), (0,9), (0, 13), (2, 5)        # Any other connections needed (middle of hand)
]

# Defining variables
mode = "file_selection"
annotate_button_pressed = False
back_button_pressed = False
drawingspace_activated = False
x_position = 0
scan_time = None
scan_stage = 0
zoom_stage = None
rotation_stage = None
file_upload_stage = 0
upload_time = None
text = ""
text_1 = ""
text_2 = ""
zoom_start_time = None
zoom_distance = 0.8
brain_file = None
brain_data = None
tumour_file = None
tumour_data = None
rotation_gesture_time = None
rotation_start_time = None
base_folder = "MRI Scans"
animation_frame = 0
scroll = 0
visible_files = 6
selected = 0
mesh_animation = False
mesh_animation_start = None
animation_duration = 6.0
menu_time = 0
draw_points = []
storing_lines = []

# Annotation variables
palette = [(0, 90, 170), (0, 0, 120), (170, 40, 0), (0, 120, 10), (120, 0, 70), (180, 180, 180)]
labels = ["Orange", "   Red", "  Blue", " Green", "Purple", "White"]
colour_index = 0
colour_name = labels[colour_index]
brush_thickness = 3
prev_x, prev_y = 0, 0
smooth_x, smooth_y = 0, 0

files = os.listdir("MRI Scans")

def hand_pressing_button(hand_landmarks,
                         x_min,
                         y_min,
                         x_max,
                         y_max,
                         w,
                         h,
                         threshold = 0.1):
    count = 0

    for landmark in hand_landmarks:
        x = int(landmark.x * w)  # Width of frame
        y = int(landmark.y * h)  # Height of frame

        if x_min <= x <= x_max and \
                y_min <= y <= y_max:
            count += 1

    return (count / 21) >= threshold


def angle(a, b, c):
    # Vector ba (ba = a-b) --> middle joint (b) to tip of finger (a) (a > b)
    ba_x, ba_y, ba_z = a.x - b.x, a.y - b.y, a.z - b.z
    # Vector bc (bc = c-b) --> middle joint (b) to base of finger (c)
    bc_x, bc_y, bc_z = c.x - b.x, c.y - b.y, c.z - b.z

    # Dot product between ba and bc (ba x bc)
    dot_product = ba_x * bc_x + ba_y * bc_y + ba_z * bc_z

    # Finding the magnitude of both vectors
    magnitude_ba = math.sqrt(ba_x ** 2 + ba_y ** 2 + ba_z ** 2)
    magnitude_bc = math.sqrt(bc_x ** 2 + bc_y ** 2 + bc_z ** 2)

    # Preventing getting zero as a denominator (which would result in error)
    if magnitude_ba == 0:
        return 0
    if magnitude_bc == 0:
        return 0

    cosine_theta = dot_product / (magnitude_ba * magnitude_bc)
    # Clamping to make sure angle stays between -1 and 1
    cosine_theta = max(min(cosine_theta, 1.0), -1.0)
    theta = math.acos(cosine_theta)
    # Putting it in degrees
    return math.degrees(theta)

def is_drawing(hand_landmarks):

    index = hand_landmarks[8]
    middle = hand_landmarks[12]

    distance = math.sqrt(
        (index.x - middle.x) ** 2 +
        (index.y - middle.y) ** 2
    )

    return distance > 0.05

# Resizing windows and positioning them
pv.global_theme.window_size =[700, 750]
cv2.namedWindow("Input Screen", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Input Screen", 578, 452)
cv2.moveWindow("Input Screen", 702, 450)
cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Camera", 578, 450)
cv2.moveWindow("Camera", 702, 0)

# DEFINING VARIABLES FOR NIBABEL

def make_brain(brain_path):
    brain_file = nib.load(brain_path)
    brain_data = brain_file.get_fdata()  # Get (f) floating point data ("Give me image data as equivalent floating point data")

    grid = pv.ImageData()
    grid.dimensions = brain_file.shape
    grid.spacing = tuple(x * 1 for x in brain_file.header.get_zooms()) # (1x1x1)mm3 voxels
    grid.origin = (0, 0, 0)
    grid.point_data["MRI"] = brain_data.flatten(order="F")
    surf = (grid.contour([100]).triangulate().flip_faces())
    return surf, brain_file

def make_tumour(tumour_path):
    tumour_file = nib.load(tumour_path)
    tumour_data = tumour_file.get_fdata()

    grid_2 = pv.ImageData()
    grid_2.dimensions = tumour_file.shape
    grid_2.spacing = tumour_file.header.get_zooms()
    grid_2.origin = (0, 0, 0)
    grid_2.point_data["Tumour"] = tumour_data.flatten(order="F")
    surf_2 = (grid_2.contour([0.5]).triangulate().flip_faces())
    return surf_2, tumour_file

def load_mri_patient(filename):
    brain_path = os.path.join("MRI Scans", filename + "-t1c.nii.gz")
    tumour_path = os.path.join("MRI Scans", filename + "-seg.nii.gz")

    # Making sure that the path exists
    if not os.path.exists(brain_path):
        print("brain path not found")
        return None, None
    if not os.path.exists(tumour_path):
        print("tumour path not found")
        return None, None

pl = pv.Plotter()
pl.set_background("black")
pl.show(auto_close=False, interactive_update=True, full_screen=True)
pl.camera_position = 'yz'
pl.camera.zoom(1.1)

# The actors are none because the brain MRI has not been chosen and loaded yet
brain_actor = None
brain_overlay_actor = None
brain_wireframe_actor = None
tumour_actor = None
tumour_overlay_actor = None
tumour_wireframe_actor = None


# WHILE TRUE LOOP


while True:

    success, img = cap.read()

    # Preventing webcam frame bug
    if not success:
        print("Error with camera frames")
        break

    h, w, c = img.shape

    frame = np.zeros((452, 578, 3), dtype=np.uint8)

    # Position overlay at (x, y)
    x, y = 0, 0

    # Draw overlay onto webcam frame
    img[y:y + overlay_flip.shape[0], x:x + overlay_flip.shape[1]] = overlay_flip

    # Flipping webcam image
    img = cv2.flip(img, 1)

    # UI design for webcam + input window
    cv2.rectangle(img, (635, 5), (485, 115), (255, 247, 164), 2)
    cv2.putText(img, str("__________"), (485, 60), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (255, 247, 164), -1)
    cv2.putText(frame, str("______________________________________________"), 
                (10, 15), cv2.FONT_HERSHEY_TRIPLEX, 0.6, (255, 247, 164), 1)

    # UI design for the input window (changing)

    if mode == "file_selection":
        cv2.putText(frame, str("CHOOSE A BraTS FILE:"), (15, 45), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (255, 247, 164), 1)
        selected_y = 90 + (selected - scroll) * 30
        cv2.rectangle(frame, (10, selected_y - 22), (400, selected_y + 5), (158, 158, 158), -1)
        cv2.rectangle(frame, (400, 68), (10, 245), (255, 247, 164), 1)

        for i in range(scroll, min(scroll + visible_files, len(files))):

            y = 90 + (i - scroll) * 30

            cv2.putText(frame, f"{i}: {files[i]}", (15, y), cv2.FONT_HERSHEY_TRIPLEX, 0.6, (255, 247, 164), 1)

    if mode == "menu" or mode == "annotate":
        cv2.putText(frame, f"{filename} selected", (128, 45), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (255, 247, 164), 1)
        patient_number = int(filename.split("-")[2])
        cv2.putText(frame, f"Patient Number: {patient_number}", 
                    (20, 72), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (255, 247, 164), 1)
        case_number = int(filename.split("-")[3])
        cv2.putText(frame, f"Case Number: {case_number}", (385, 72), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (255, 247, 164), 1)
        cv2.putText(frame, str("______________________________________________"), 
                    (10, 86), cv2.FONT_HERSHEY_TRIPLEX, 0.6, (255, 247, 164), 1)
        frame[90:290, 10:410] = manual
        cv2.rectangle(frame, (420, 100), (565, 260), (255, 247, 164), 1)
        cv2.putText(frame, str("Press"), (455, 130), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 247, 164), -1)
        cv2.putText(frame, str("r"), (484, 160), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 247, 164), -1)
        cv2.putText(frame, str("to select"), (435, 190), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 247, 164), -1)
        cv2.putText(frame, str("another"), (439, 220), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 247, 164), -1)
        cv2.putText(frame, str("patient"), (444, 252), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 247, 164), -1)

    darkness = 0.55

    img = (img * darkness).astype("uint8")

    # Converting colour channels from RGB to BGR
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Makes format of image readable to mediapipe
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_img
    )

    # Creating timestamps in milliseconds for stable tracking between frames.
    timestamp_ms = int(time.time() * 1000)

    # Sends current image frame and timestamp to the hand-tracking model.
    result = detector.detect_for_video(mp_image, timestamp_ms)


# MESH ANIMATION

    if mesh_animation == True and mode == "file_selection":

        passing_time = time.time() - mesh_animation_start

        progress = passing_time / animation_duration

        progress = max(0.0, min(1.0, progress))

        progress = progress ** 2

        opacity_brain = 0.30 * progress
        opacity_brain_wireframe = 0.35 * progress
        fade_in_variable = 1.0 * progress
        opacity_tumour_overlay = 0.85 * progress
        opacity_tumour_wireframe = 0.9 * progress

        brain_actor.GetProperty().SetOpacity(min(opacity_brain, 0.30))
        brain_actor.GetProperty().SetAmbient(min(fade_in_variable, 1.0))
        brain_overlay_actor.GetProperty().SetAmbient(min(fade_in_variable, 1.0))
        brain_wireframe_actor.GetProperty().SetOpacity(min(opacity_brain_wireframe, 0.35))
        tumour_actor.GetProperty().SetOpacity(min(fade_in_variable, 1.0))
        tumour_actor.GetProperty().SetAmbient(min(fade_in_variable, 1.0))
        tumour_overlay_actor.GetProperty().SetOpacity(min(opacity_tumour_overlay, 0.85))
        tumour_overlay_actor.GetProperty().SetAmbient(min(fade_in_variable, 1.0))
        tumour_wireframe_actor.GetProperty().SetOpacity(min(opacity_tumour_wireframe, 0.9))
        tumour_wireframe_actor.GetProperty().SetAmbient(min(fade_in_variable, 1.0))

    if result.hand_landmarks:

        for hand_landmarks in result.hand_landmarks:

            # Draw lines between dots (UX extra feature)
            for start_idx, end_idx in hand_lines:

                start = hand_landmarks[start_idx]
                end = hand_landmarks[end_idx]

                x1, y1 = int(start.x * w), int(start.y * h)
                x2, y2 = int(end.x * w), int(end.y * h)
                # Main colour layer for the lines (BGR)
                cv2.line(img, (x1, y1), (x2, y2), (255, 247, 164), 1)

                for landmark in hand_landmarks:

                    x = int(landmark.x * w)  # Width of frame
                    y = int(landmark.y * h)  # Height of frame

                    # Main colour for the landmarks
                    cv2.circle(img, (x, y), 4, (255, 247, 164), -1)


        # Just look at the angles for the fingers to determine if they are bent or not.
        # Reason = finger angles, unlike distances, will not change as the z-axis changes (depth).
        for i, hand_landmarks in enumerate(result.hand_landmarks):

                if not result.handedness or i >= len(result.handedness):
                    continue

        finger_state = []
        type_text = ""

        for finger_name, (base, pip, top) in fingers.items():

            pip_angle = angle(hand_landmarks[base],
                                hand_landmarks[pip],
                                hand_landmarks[top])

            # Checking to see if it works
            #if pip_angle < 160:
            #    print(f"{finger_name}: BENT")
            #else:
            #    print(f"{finger_name}: STRAIGHT")

            if pip_angle < 160:
                finger_state.append(0)
            else:
                finger_state.append(1)

# FEATURES (ZOOM, ROTATE, AUTO-ROTATE AND HIGHLIGHT TUMOUR MODE)

        # Zoom and rotate feature will work on all modes,
        # any other hand detection feature will be strictly only for the menu mode.

        if mode == "menu":

            # PINCH GESTURE - Zoom in and out of brain model
            if finger_state == [1, 0, 1, 1, 1]:

                if zoom_start_time is None:
                    zoom_start_time = time.time()  # (Detecting the gesture on screen for the first time)
                elif time.time() - zoom_start_time >= 1.5:
                    zoom_stage = 1

            if zoom_stage == 1 and rotation_stage != 1:
                    cv2.putText(img, str("ZOOM MODE"), (505, 30), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (255, 247, 164), 1)
                    cv2.putText(img, str("ACTIVATED"), (508, 50), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (255, 247, 164), 1)

                    x1 = int(hand_landmarks[4].x * w)   # Hand landmark 4 is the thumb
                    y1 = int(hand_landmarks[4].y * h)
                    x2 = int(hand_landmarks[8].x * w)   # Hand landmark 8 is the index finger
                    y2 = int(hand_landmarks[8].y * h)

                    cv2.line(img, (x1, y1), (x2, y2), (255, 247, 164), 3)
                    cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 1)

                    length = math.hypot(x2 - x1, y2 - y1)
                    norm_length = (length - 14)/ (228 - 14)  # Normalising the length so the value is not always 1
                    # (14 was the minimum  value in the print results and 228 was the maximum value)
                    zoom_value = max(0.0, min(norm_length, 1.0))
                    zoom_speed = (zoom_value - 0.3) * 0.09  # Slowing down zoom as it was too quick before
                    pl.camera.zoom(1.0 + zoom_speed)

            # PINKY AND THUMB GESTURE - manual rotation mode
            if finger_state == [1, 0, 0, 0, 1] and zoom_stage is None:

                if rotation_start_time is None:
                    rotation_start_time = time.time() # (Detecting the gesture on screen for the first time)
                elif time.time() - rotation_start_time >= 1.5:
                    rotation_stage = 1

            # ROCK GESTURE - Automatic rotation mode
            if finger_state == [1, 1, 0, 0, 1]:

                if rotation_start_time is None:
                    rotation_start_time = time.time()  # (Detecting the gesture on screen for the first time)
                elif time.time() - rotation_start_time >= 1.5:
                    rotation_stage = 2

            if rotation_stage == 1:
                cv2.putText(img, str("MANUAL ROTATION"), (493, 78), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 247, 164), -1)
                cv2.putText(img, str("ACTIVATED"), (508, 102), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (255, 247, 164), -1)
                cv2.line(img, (92, 400), (438, 200), (0, 0, 0), 2)
                cv2.line(img, (92, 200), (438, 400), (0, 0, 0), 2)
                cv2.line(img, (172, 456), (358, 144), (0, 0, 0), 2)
                cv2.line(img, (172, 144), (358, 456), (0, 0, 0), 2)
                cv2.putText(img, str("x-axis"), (370, 300), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 0, 0), -1)
                cv2.putText(img, str("anticlockwise"), (340, 320), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 0, 0), -1)
                cv2.putText(img, str("x-axis"), (120, 300), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 0, 0), -1)
                cv2.putText(img, str("clockwise"), (100, 320), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 0, 0), -1)
                cv2.putText(img, str("z-axis"), (335, 212), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 0, 0), -1)
                cv2.putText(img, str("z-axis"), (335, 394), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 0, 0), -1)
                cv2.putText(img, str("z-axis"), (140, 212), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 0, 0), -1)
                cv2.putText(img, str("z-axis"), (140, 394), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 0, 0), -1)
                cv2.putText(img, str("y-axis"), (240, 165), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 0, 0), -1)
                cv2.putText(img, str("upwards"), (229, 186), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 0, 0), -1)
                cv2.putText(img, str("y-axis"), (240, 425), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 0, 0), -1)
                cv2.putText(img, str("downwards"), (219, 445), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 0, 0), -1)

                x1 = int(hand_landmarks[8].x * w)  # Hand landmark 8 is the index finger
                y1 = int(hand_landmarks[8].y * h)

                cv2.circle(img, (265, 300), 9, (0, 0, 0), -1)
                cv2.line(img, (x1, y1), (265, 300), (255, 247, 164), 3)

                length = math.hypot(265 - x1, 300 - y1)
                norm_length = length * 0.03
                rotation_speed = norm_length  # Distance of line = speed of rotation

                # Work out the angle to see if it is a y, z or x rotation
                dx = x1 - 265
                dy = y1 - 300

                angle_check = math.degrees(math.atan2(-dy, dx))

                #cv2.putText(img, str(angle_check), (200, 200), cv2.FONT_HERSHEY_TRIPLEX, 1, (0,0,0), 1)
                #cv2.putText(img, str(y1), (200, 200), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 0), 1)

                # No rotation to start with
                rotation_y_direction = 0
                rotation_z_direction = 0
                rotation_x_direction = 0

                # X-axis rotation
                if x1 <= 265 and (-180 <= angle_check <= -150 or 150 <= angle_check <= 180):
                    # Where index finger is on the screen determines if it is spinning clockwise or anticlockwise
                    # along the x-axis
                    rotation_x_direction = -1

                elif x1 > 265 and -30 <= angle_check <= 30:
                    rotation_x_direction = 1

                # Z-axis rotation (still based on x-axis values but new angles)
                elif x1 <= 265 and (-140 <= angle_check <= -125 or 125 <= angle_check <= 140):
                    rotation_z_direction = -1

                elif x1 > 265 and (-55 <= angle_check <= -40 or 40 <= angle_check <= 55):
                    rotation_z_direction = 1

                # Y-axis rotation
                elif y1 > 300 and (-115 < angle_check < -60):
                    rotation_y_direction = 1

                elif y1 <= 300 and (115 > angle_check > 60):
                    rotation_y_direction = -1

                # Using camera rotations so that the model does not go off-screen
                pl.camera.Azimuth(rotation_x_direction * (1.0 + rotation_speed))
                y_axis_rotation = rotation_y_direction * (0.25 * rotation_speed)
                pl.camera.Elevation(y_axis_rotation)
                z_axis_rotation = max(-50, min(50, rotation_z_direction * (0.6 * rotation_speed)))
                pl.camera.Roll(z_axis_rotation)

            if rotation_stage == 2:
                cv2.putText(img, str("AUTO-ROTATE"), (513, 78), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 247, 164), -1)
                cv2.putText(img, str("ACTIVATED"), (508, 102), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (255, 247, 164), -1)
                pl.camera.Azimuth(1.2)

            # FIST GESTURE - To end the zoom or rotation activation
            if finger_state == [0, 0, 0, 0, 0]:
                zoom_stage = None
                zoom_start_time = None  # (If the gesture has disappeared)
                rotation_stage = None
                rotation_start_time = None

            # PEACE GESTURE - Highlight the tumour only
            if finger_state == [0, 1, 1, 0, 0]:

                if animation_frame < 40:
                    animation_frame += 1

                tumour_mesh_value.SetVisibility(True)
                tumour_value.SetVisibility(True)
                percentage_brain.SetVisibility(False)
                percentage_tumour.SetVisibility(True)
                brain_mesh_value.SetVisibility(False)
                brain_value.SetVisibility(False)

            # THREE FINGERS HELD UP - Brain + tumour mode
            elif finger_state == [0, 1, 1, 1, 0]:

                animation_frame -= 1

                tumour_mesh_value.SetVisibility(False)
                tumour_value.SetVisibility(False)
                percentage_brain.SetVisibility(True)
                percentage_tumour.SetVisibility(False)
                brain_mesh_value.SetVisibility(True)
                brain_value.SetVisibility(True)

            progress = max(0.0, min(1.0, animation_frame / 40.0))

            opacity_brain = 0.30 - (0.30 * progress)
            ambient_brain = 1.0 - (1.0 * progress)
            diffuse_brain = 0.05 - (0.05 * progress)
            specular_brain = 0.8 - (0.8 * progress)
            sp_brain = 50 - (50 * progress)
            specular_overlay = 1.0 - (1.0 * progress)
            opacity_wireframe = 0.35 - (0.35 * progress)

            brain_actor.GetProperty().SetOpacity(opacity_brain)
            brain_actor.GetProperty().SetAmbient(ambient_brain)
            brain_actor.GetProperty().SetDiffuse(diffuse_brain)
            brain_actor.GetProperty().SetSpecular(specular_brain)
            brain_actor.GetProperty().SetSpecularPower(sp_brain)
            brain_overlay_actor.GetProperty().SetAmbient(ambient_brain)
            brain_overlay_actor.GetProperty().SetSpecular(specular_overlay)
            brain_wireframe_actor.GetProperty().SetOpacity(opacity_wireframe)

# BUTTONS

    if mode == "menu":

        # Annotate button dimensions
        abutton_x_min = 480
        abutton_x_max = 635
        abutton_y_min = 140
        abutton_y_max = 200

        cv2.rectangle(img, (635, 140), (480, 200), (0, 0, 0), -1)
        cv2.rectangle(img, (635, 140), (480, 200), (255, 247, 164), 1)
        cv2.putText(img, str("ANNOTATE"), (482, 180), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 247, 164), 1)

# PRESSING THE BUTTON (BUTTON DETECTION)

        if result.hand_landmarks:

            hand_over_abutton = False

            for hand_landmarks in result.hand_landmarks:

                # Hand over annotate button
                if hand_pressing_button(hand_landmarks,
                                        abutton_x_min, abutton_y_min,
                                        abutton_x_max, abutton_y_max,
                                        w, h):
                    hand_over_abutton = True

            if hand_over_abutton and not annotate_button_pressed:
                mode = "annotate"
                annotate_button_pressed = True

            if not hand_over_abutton:
                annotate_button_pressed = False

# THE ANNOTATE FEATURE

    if mode == "annotate":

        # Back button dimensions + landmark detection
        backbutton_x_min = 560
        backbutton_x_max = 642
        backbutton_y_min = 440
        backbutton_y_max = 478

        cv2.rectangle(img, (642, 440), (560, 478), (0, 0, 0), -1)
        cv2.rectangle(img, (642, 440), (560, 478), (255, 247, 164), 1)
        cv2.putText(img, str("BACK"), (562, 468), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 247, 164), 1)

        # UI design for the annotation feature
        drawingspace_x_min = 10
        drawingspace_x_max = 478
        drawingspace_y_min = 178
        drawingspace_y_max = 470

        cv2.rectangle(img, (478, 178), (10, 470), (0, 0, 0), 1)

        if result.hand_landmarks:

            hand_over_backbutton = False
            hand_in_drawingspace = False

            for hand_landmarks in result.hand_landmarks:
                if hand_pressing_button(hand_landmarks,
                                        backbutton_x_min, backbutton_y_min,
                                        backbutton_x_max, backbutton_y_max,
                                        w, h):
                    hand_over_backbutton = True

                if hand_pressing_button(hand_landmarks,
                                        drawingspace_x_min, drawingspace_y_min,
                                        drawingspace_x_max, drawingspace_y_max,
                                        w, h):
                    hand_in_drawingspace = True

            if hand_over_backbutton and not back_button_pressed:
                mode = "menu"
                back_button_pressed = True

                # Clearing all lines drawn
                for actors in storing_lines:
                    pl.remove_actor(actors)

                storing_lines.clear()
                draw_points = []
                prev_x, prev_y = 0, 0

            if not hand_over_backbutton:
                back_button_pressed = False

            if hand_in_drawingspace and not drawingspace_activated:
                drawingspace_activated = True

            if not hand_in_drawingspace:
                drawingspace_activated = False

        # Making the annotation colour selection feature
        for i, colour in enumerate(palette):
            x_start = 20 + (i * 80)
            x_end = x_start + 45
            cv2.rectangle(img, (x_start, 124), (x_end, 160), colour, -1)
            cv2.rectangle(img, (x_start, 124), (x_end, 160), (0, 0, 0), 1)
            cv2.putText(img, labels[i], (x_start, 175), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 0), 1)

        if result.hand_landmarks:
            for hand_landmarks in result.hand_landmarks:
                lm_list = []
                for id, lm in enumerate(hand_landmarks):
                    lm_list.append((int(lm.x * w), int(lm.y * h)))

                index_x, index_y = lm_list[8]

                cv2.circle(img, (index_x, index_y), 8, (0, 0, 0), -1)

                # Selecting the colour on the colour palette
                if 124 <= index_y <= 160:
                    colour_index = min((index_x - 20)// 80, len(palette) - 1)
                    colour_name = labels[colour_index]

                    cv2.circle(img, (index_x, index_y), 6, palette[colour_index], -1)

                if drawingspace_activated:

                    if prev_x == 0 and prev_y == 0:
                        prev_x, prev_y = index_x, index_y

                    else:

                        cv2.line(img, (prev_x, prev_y), (index_x, index_y), palette[colour_index], brush_thickness)
                        cv2.putText(img, colour_name, (501, 42), cv2.FONT_HERSHEY_TRIPLEX, 1.2, palette[colour_index], -1)
                        cv2.putText(img, str("selected"), (490, 98), cv2.FONT_HERSHEY_TRIPLEX, 1.2, palette[colour_index], -1)

                        prev_x, prev_y = index_x, index_y

                if drawingspace_activated and is_drawing(hand_landmarks):
                    x = ((index_x - drawingspace_x_min) /
                         (drawingspace_x_max - drawingspace_x_min)) * 300

                    y = (1 - ((index_y - drawingspace_y_min) /
                              (drawingspace_y_max - drawingspace_y_min))) * 200

                    if smooth_x == 0 or smooth_y == 0:
                        smooth_x, smooth_y = x, y
                    else:
                        smooth_x = smooth_x * 0.7 + x * 0.2
                        smooth_y = smooth_y * 0.7 + y * 0.2

                    point = (250, smooth_x, smooth_y)

                    draw_points.append(point)

                    line = pv.lines_from_points(np.array(draw_points))

                    # (Reverse the colour order because PyVista is RGB and OpenCV is BGR)
                    pyvista_colour = palette[colour_index][::-1]

                    line_actor = pl.add_mesh(line, color=pyvista_colour, line_width=4)

                    storing_lines.append(line_actor)

                else:

                    draw_points = []

# DISPLAY + KEY FEATURES

    cv2.imshow("Camera", img)
    cv2.imshow("Input Screen", frame)
    pl.update()
    key = cv2.waitKey(1)

    # Press q to end the loop
    if key == ord('q'):
        break

    if key == ord("s"):
        selected = max(0, selected + 1)

        if selected >= scroll + visible_files:
            scroll += 1

    if key == ord("w"):
        selected = max(0, min(len(files) - visible_files, selected - 1))

        if selected < scroll:
            scroll -= 1

    if key == 13 and mode == "file_selection":
        filename = files[selected]
        brain_filename = filename + "-t1c.nii.gz"
        tumour_filename = filename + "-seg.nii.gz"
        brain_path = os.path.join("MRI Scans", filename, brain_filename)
        tumour_path = os.path.join("MRI Scans", filename, tumour_filename)

        brain_mesh, brain_file = make_brain(brain_path)
        brain_volume = brain_mesh.volume # (In mm^3)
        brain_volume_cm3 = brain_volume / 1000
        tumour_mesh, tumour_file = make_tumour(tumour_path)
        tumour_volume = tumour_mesh.volume # (In mm^3)
        tumour_volume_cm3 = tumour_volume / 1000

        estimated_brain_volume = np.sum(brain_file.get_fdata() > 0)
        estimated_brain_volume_cm3 = estimated_brain_volume / 1000
        estimated_tumour_volume = np.sum(tumour_file.get_fdata() > 0)
        estimated_tumour_volume_cm3 = estimated_tumour_volume / 1000

        # Load the meshes
        brain_actor = pl.add_mesh(
            brain_mesh,
            color="#00BFFF",
            opacity=0.30,
            ambient=1.0,
            diffuse=0.05,
            specular=0.8,
            specular_power=50,
            smooth_shading=True
        )

        brain_overlay_actor = pl.add_mesh(
            brain_mesh,
            color="#BDF5FF",
            opacity=0.08,
            ambient=1.0,
            diffuse=0.0,
            specular=1.0,
            smooth_shading=True
        )

        brain_wireframe_actor = pl.add_mesh(
            brain_mesh,
            color="#7FEFFF",
            opacity=0.35,
            style="wireframe",
            line_width=0.7
        )

        tumour_actor = pl.add_mesh(
            tumour_mesh,
            color="#FF0000",
            opacity=1.0,
            ambient=1.0,
            diffuse=0.05,
            specular=0.9,
            specular_power=50,
            smooth_shading=True,
            split_sharp_edges=True
        )

        tumour_overlay_actor = pl.add_mesh(
            tumour_mesh,
            color="#FF3030",
            opacity=0.85,
            ambient=1.0,
            diffuse=0.02,
            specular=1.0,
            smooth_shading=True,
            split_sharp_edges=True
        )

        tumour_wireframe_actor = pl.add_mesh(
            tumour_mesh,
            color="#FF8080",
            opacity=0.9,
            ambient=1.0,
            diffuse=0.05,
            specular=0.5,
            style="wireframe",
            line_width=1.0,
            smooth_shading=True,
            split_sharp_edges=True
        )

        brain_mesh_value = pl.add_text(f"Brain mesh size = {brain_volume_cm3: .2f} cm³", position=(15, 708), font_size=14, font="times",
                                       color="#A4F7FF")
        brain_value = pl.add_text(f"Estimated brain size = {estimated_brain_volume_cm3: .2f} cm³", position=(13, 672), font_size=14, font="times",
                                  color="#A4F7FF")
        percentage_difference = (abs(brain_volume_cm3 - estimated_brain_volume_cm3)/estimated_brain_volume_cm3) * 100
        percentage_brain = pl.add_text(f"(% difference = {percentage_difference: .3f}%)", position=(14, 648),
                                       font_size=9, font="times", color="#A4F7FF")
        tumour_mesh_value = pl.add_text(f"Tumour mesh size = {tumour_volume_cm3: .2f} cm³", position=(15, 708), font_size=14, font="times",
                                        color="#A4F7FF")
        tumour_value = pl.add_text(f"Estimated tumour size = {estimated_tumour_volume_cm3: .2f} cm³", position=(13, 672), font_size=14, font="times",
                                   color="#A4F7FF")
        percentage_difference_2 = (abs(tumour_volume_cm3 - estimated_tumour_volume_cm3) / estimated_tumour_volume_cm3) * 100
        percentage_tumour = pl.add_text(f"(% difference = {percentage_difference_2: .3f}%)", position=(14, 648),
                                       font_size=9, font="times", color="#A4F7FF")

        tumour_mesh_value.SetVisibility(False)
        tumour_value.SetVisibility(False)
        percentage_brain.SetVisibility(True)
        percentage_tumour.SetVisibility(False)
        brain_mesh_value.SetVisibility(True)
        brain_value.SetVisibility(True)

        pl.reset_camera()

        mesh_animation = True
        mesh_animation_start = time.time()

        # After loading the meshes
        mode = "menu"

    if key == ord("r") and mode != "file_selection":
        pl.remove_actor(brain_actor)
        pl.remove_actor(brain_overlay_actor)
        pl.remove_actor(brain_wireframe_actor)
        pl.remove_actor(tumour_actor)
        pl.remove_actor(tumour_overlay_actor)
        pl.remove_actor(tumour_wireframe_actor)

        tumour_mesh_value.SetVisibility(False)
        tumour_value.SetVisibility(False)
        percentage_brain.SetVisibility(False)
        percentage_tumour.SetVisibility(False)
        brain_mesh_value.SetVisibility(False)
        brain_value.SetVisibility(False)

        mode = "file_selection"

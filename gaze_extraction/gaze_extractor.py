import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils, drawing_styles
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as Rscipy
from collections import deque
import pyautogui
import math

# === Screen mapping config ===
MONITOR_WIDTH, MONITOR_HEIGHT = pyautogui.size()
calibration_offset_yaw = 0
calibration_offset_pitch = 0

# === Nose landmark indices for stable head tracking ===
nose_indices = [4, 45, 275, 220, 440, 1, 5, 51, 281, 44, 274, 241,
                461, 125, 354, 218, 438, 195, 167, 393, 165, 391,
                3, 248]

# === Eye sphere calibration state ===
left_sphere_locked = False
left_sphere_local_offset = None
right_sphere_locked = False
right_sphere_local_offset = None

# === Scale-aware tracking ===
calibration_nose_scale = None  # nose scale at calibration time
BASE_EYE_RADIUS = 20  # approximate eyeball radius in frame units at calibration distance

# === PCA reference matrix (prevents axis flipping between frames) ===
R_ref = [None]

# === Smoothing buffer ===
SMOOTH_LENGTH = 10
gaze_buffer = deque(maxlen=SMOOTH_LENGTH)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def compute_nose_scale(landmarks, w, h):
    """Average pairwise distance of nose landmarks — used for scale tracking."""
    points = np.array([
        [landmarks[i].x * w, landmarks[i].y * h, landmarks[i].z * w]
        for i in nose_indices
    ])
    n = len(points)
    total = 0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += np.linalg.norm(points[i] - points[j])
            count += 1
    return total / count if count > 0 else 1.0


def gaze_to_screen(gaze_direction, offset_yaw, offset_pitch):
    """Convert 3D gaze vector to screen coordinates via yaw/pitch angles."""
    d = gaze_direction / np.linalg.norm(gaze_direction)

    # Yaw (horizontal angle from forward)
    xz = np.array([d[0], 0, d[2]])
    xz /= np.linalg.norm(xz)
    yaw = math.degrees(math.acos(np.clip(np.dot([0, 0, -1], xz), -1, 1)))
    if d[0] < 0:
        yaw = -yaw

    # Pitch (vertical angle from forward)
    yz = np.array([0, d[1], d[2]])
    yz /= np.linalg.norm(yz)
    pitch = math.degrees(math.acos(np.clip(np.dot([0, 0, -1], yz), -1, 1)))
    if d[1] > 0:
        pitch = -pitch

    # Flip yaw (camera is mirrored)
    yaw = -yaw

    # Apply calibration offsets
    yaw += offset_yaw
    pitch += offset_pitch

    # Degrees of eye rotation that reach screen edges — tune these
    yaw_range = 15.0
    pitch_range = 10.0

    # Map to screen pixels
    sx = int(((yaw + yaw_range) / (2 * yaw_range)) * MONITOR_WIDTH)
    sy = int(((pitch_range - pitch) / (2 * pitch_range)) * MONITOR_HEIGHT)

    sx = max(0, min(sx, MONITOR_WIDTH - 1))
    sy = max(0, min(sy, MONITOR_HEIGHT - 1))

    return sx, sy, yaw, pitch


def get_head_pose(landmarks, w, h, r_ref):
    """PCA-based head center and orientation from nose landmarks."""
    points_3d = np.array([
        [landmarks[i].x * w, landmarks[i].y * h, landmarks[i].z * w]
        for i in nose_indices
    ])
    center = np.mean(points_3d, axis=0)

    centered = points_3d - center
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvecs = eigvecs[:, np.argsort(-eigvals)]

    if np.linalg.det(eigvecs) < 0:
        eigvecs[:, 2] *= -1

    R_final = Rscipy.from_matrix(eigvecs).as_matrix()

    # Stabilize against sign flips
    if r_ref[0] is None:
        r_ref[0] = R_final.copy()
    else:
        for i in range(3):
            if np.dot(R_final[:, i], r_ref[0][:, i]) < 0:
                R_final[:, i] *= -1

    return center, R_final


def get_iris_3d(landmarks, idx, w, h):
    """Get iris landmark as a 3D point in frame coordinates."""
    lm = landmarks[idx]
    return np.array([lm.x * w, lm.y * h, lm.z * w])


def draw_landmarks_on_image(rgb_image, detection_result):
    face_landmarks_list = detection_result.face_landmarks
    annotated_image = np.copy(rgb_image)

    for idx in range(len(face_landmarks_list)):
        face_landmarks = face_landmarks_list[idx]

        drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks,
            connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=drawing_styles.get_default_face_mesh_tesselation_style())
        drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks,
            connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=drawing_styles.get_default_face_mesh_contours_style())
        drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks,
            connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_LEFT_IRIS,
            landmark_drawing_spec=None,
            connection_drawing_spec=drawing_styles.get_default_face_mesh_iris_connections_style())
        drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks,
            connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_IRIS,
            landmark_drawing_spec=None,
            connection_drawing_spec=drawing_styles.get_default_face_mesh_iris_connections_style())

    return annotated_image


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    global left_sphere_locked, left_sphere_local_offset
    global right_sphere_locked, right_sphere_local_offset
    global R_ref
    global calibration_offset_yaw, calibration_offset_pitch
    global calibration_nose_scale, gaze_buffer

    # Create the FaceLandmarker
    base_options = python.BaseOptions(model_asset_path='face_landmarker_v2_with_blendshapes.task')
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
        num_faces=1
    )
    detector = vision.FaceLandmarker.create_from_options(options)

    # Open webcam
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = detector.detect(mp_image)

        annotated_image = draw_landmarks_on_image(rgb_frame, detection_result)
        bgr_annotated = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)

        if detection_result.face_landmarks:
            landmarks = detection_result.face_landmarks[0]
            h, w = frame.shape[:2]

            head_center, R_head = get_head_pose(landmarks, w, h, R_ref)

            iris_3d_left = get_iris_3d(landmarks, 468, w, h)
            iris_3d_right = get_iris_3d(landmarks, 473, w, h)

            if left_sphere_locked and right_sphere_locked:
                # --- Scale-aware sphere reconstruction ---
                current_nose_scale = compute_nose_scale(landmarks, w, h)
                scale_ratio = current_nose_scale / calibration_nose_scale if calibration_nose_scale else 1.0

                sphere_l = head_center + R_head @ (left_sphere_local_offset * scale_ratio)
                sphere_r = head_center + R_head @ (right_sphere_local_offset * scale_ratio)

                # Gaze direction = iris - sphere center
                left_gaze = iris_3d_left - sphere_l
                left_gaze /= np.linalg.norm(left_gaze)
                right_gaze = iris_3d_right - sphere_r
                right_gaze /= np.linalg.norm(right_gaze)

                raw_combined = (left_gaze + right_gaze) / 2
                raw_combined /= np.linalg.norm(raw_combined)

                # --- Smoothing buffer ---
                gaze_buffer.append(raw_combined)
                combined_gaze = np.mean(gaze_buffer, axis=0)
                combined_gaze /= np.linalg.norm(combined_gaze)

                # --- Draw per-eye gaze rays (thin green) ---
                gaze_length = 200
                for sphere, iris in [(sphere_l, iris_3d_left), (sphere_r, iris_3d_right)]:
                    direction = iris - sphere
                    direction /= np.linalg.norm(direction)
                    endpoint = sphere + direction * gaze_length
                    cv2.line(bgr_annotated,
                             (int(sphere[0]), int(sphere[1])),
                             (int(endpoint[0]), int(endpoint[1])),
                             (0, 255, 0), 1)

                # --- Draw combined gaze ray (thick yellow) ---
                combined_origin = (sphere_l + sphere_r) / 2
                combined_endpoint = combined_origin + combined_gaze * gaze_length
                cv2.line(bgr_annotated,
                         (int(combined_origin[0]), int(combined_origin[1])),
                         (int(combined_endpoint[0]), int(combined_endpoint[1])),
                         (0, 255, 255), 2)

                # --- Screen mapping ---
                screen_x, screen_y, raw_yaw, raw_pitch = gaze_to_screen(
                    combined_gaze, calibration_offset_yaw, calibration_offset_pitch)

                # --- HUD text ---
                cv2.putText(bgr_annotated,
                            f"Gaze: ({combined_gaze[0]:.2f}, {combined_gaze[1]:.2f}, {combined_gaze[2]:.2f})",
                            (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(bgr_annotated, f"Screen: ({screen_x}, {screen_y})",
                            (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(bgr_annotated, f"Scale: {scale_ratio:.2f}",
                            (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            else:
                cv2.putText(bgr_annotated, "Press 'c' to calibrate (look at camera)",
                            (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow('Face Landmarks', bgr_annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

        elif key == ord('c') and detection_result.face_landmarks:
            # --- Eye sphere calibration ---
            landmarks = detection_result.face_landmarks[0]
            h, w = frame.shape[:2]
            head_center, R_head = get_head_pose(landmarks, w, h, R_ref)

            iris_3d_left = get_iris_3d(landmarks, 468, w, h)
            iris_3d_right = get_iris_3d(landmarks, 473, w, h)

            # Store iris position in head-local coordinates
            left_sphere_local_offset = R_head.T @ (iris_3d_left - head_center)
            right_sphere_local_offset = R_head.T @ (iris_3d_right - head_center)

            # Push sphere center back behind the iris (like a real eyeball)
            camera_dir_local = R_head.T @ np.array([0, 0, 1])
            left_sphere_local_offset += BASE_EYE_RADIUS * camera_dir_local
            right_sphere_local_offset += BASE_EYE_RADIUS * camera_dir_local

            # Store nose scale at calibration for distance tracking
            calibration_nose_scale = compute_nose_scale(landmarks, w, h)

            left_sphere_locked = True
            right_sphere_locked = True

            # Clear smoothing buffer on recalibration
            gaze_buffer.clear()

            print(f"[Calibrated] Eye spheres locked. Nose scale: {calibration_nose_scale:.2f}")

        elif key == ord('s') and left_sphere_locked and right_sphere_locked:
            # --- Screen center calibration ---
            landmarks = detection_result.face_landmarks[0]
            h, w = frame.shape[:2]
            head_center, R_head = get_head_pose(landmarks, w, h, R_ref)
            iris_l = get_iris_3d(landmarks, 468, w, h)
            iris_r = get_iris_3d(landmarks, 473, w, h)

            current_nose_scale = compute_nose_scale(landmarks, w, h)
            scale_ratio = current_nose_scale / calibration_nose_scale if calibration_nose_scale else 1.0

            sphere_l = head_center + R_head @ (left_sphere_local_offset * scale_ratio)
            sphere_r = head_center + R_head @ (right_sphere_local_offset * scale_ratio)

            lg = (iris_l - sphere_l)
            lg /= np.linalg.norm(lg)
            rg = (iris_r - sphere_r)
            rg /= np.linalg.norm(rg)
            cg = (lg + rg) / 2
            cg /= np.linalg.norm(cg)

            _, _, raw_yaw, raw_pitch = gaze_to_screen(cg, 0, 0)
            calibration_offset_yaw = -raw_yaw
            calibration_offset_pitch = -raw_pitch
            print(f"[Screen Calibrated] yaw offset: {calibration_offset_yaw:.2f}, pitch offset: {calibration_offset_pitch:.2f}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh

def extract_key_landmarks(landmarks, image_shape):
    h, w = image_shape[:2]
    def get_point(index):
        lm = landmarks.landmark[index]
        return np.array([int(lm.x * w), int(lm.y * h)], dtype=np.float32)

    return {
        "left_eye": get_point(33),
        "right_eye": get_point(263),
        "nose_tip": get_point(1)
    }

def align_static_face(static_image_path, reference_frame):
    # Load static image
    static_image = cv2.imread(static_image_path)
    static_image_rgb = cv2.cvtColor(static_image, cv2.COLOR_BGR2RGB)

    # Detect landmarks in static image
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1) as face_mesh:
        static_results = face_mesh.process(static_image_rgb)
        if not static_results.multi_face_landmarks:
            raise ValueError("No face detected in static image.")
        static_landmarks = extract_key_landmarks(static_results.multi_face_landmarks[0], static_image.shape)

    # Detect landmarks in reference frame
    reference_rgb = cv2.cvtColor(reference_frame, cv2.COLOR_BGR2RGB)
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1) as face_mesh:
        ref_results = face_mesh.process(reference_rgb)
        if not ref_results.multi_face_landmarks:
            raise ValueError("No face detected in reference frame.")
        ref_landmarks = extract_key_landmarks(ref_results.multi_face_landmarks[0], reference_frame.shape)

    # Compute affine transform
    src_points = np.array([
        static_landmarks["left_eye"],
        static_landmarks["right_eye"],
        static_landmarks["nose_tip"]
    ], dtype=np.float32)

    dst_points = np.array([
        ref_landmarks["left_eye"],
        ref_landmarks["right_eye"],
        ref_landmarks["nose_tip"]
    ], dtype=np.float32)

    M = cv2.getAffineTransform(src_points, dst_points)

    # Warp static image to match reference frame
    aligned_face = cv2.warpAffine(static_image, M, (reference_frame.shape[1], reference_frame.shape[0]))
    preview_size = (300, 300)
    aligned_face_resized = cv2.resize(aligned_face, preview_size)


    return aligned_face_resized
import cv2
import numpy as np

def blend_faces(base_frame, aligned_face):
    # Resize aligned face to match base frame
    aligned_face = cv2.resize(aligned_face, (base_frame.shape[1], base_frame.shape[0]))

    # Create a mask for the aligned face (white where face exists)
    mask = 255 * np.ones(aligned_face.shape, aligned_face.dtype)

    # Define center for cloning (e.g., center of the frame)
    center = (base_frame.shape[1] // 2, base_frame.shape[0] // 2)

    # Perform seamless cloning
    blended = cv2.seamlessClone(aligned_face, base_frame, mask, center, cv2.NORMAL_CLONE)
    return blended
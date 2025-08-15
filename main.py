from modules.face_swap import detect_face_landmarks
from modules.face_alignment import align_static_face
from modules.face_swapper import blend_faces
import cv2
import os

video_path = "assets/4587767-uhd_3840_2160_25fps.mp4"
static_image_path = "assets/cutface.png.png"
output_dir = "outputs/swapped_frames"
os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
frame_id = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    try:
        aligned_face = align_static_face(static_image_path, frame)
        swapped_frame = blend_faces(frame, aligned_face)

        cv2.imshow("Swapped Frame", swapped_frame)
        cv2.imwrite(f"{output_dir}/frame_{frame_id:04d}.png", swapped_frame)
        frame_id += 1

    except Exception as e:
        print(f"Swap failed: {e}")
        continue

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
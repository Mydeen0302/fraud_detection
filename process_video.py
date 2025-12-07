# process_video.py
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import argparse
import os

mp_face_mesh = mp.solutions.face_mesh

LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]
LEFT_IRIS_IDX = 468
RIGHT_IRIS_IDX = 473

def eye_aspect_ratio(landmarks, eye_points):
    pts = np.array([landmarks[i] for i in eye_points], dtype=float)
    v1 = np.linalg.norm(pts[1] - pts[5])
    v2 = np.linalg.norm(pts[2] - pts[4])
    vert = (v1 + v2)/2.0
    horiz = np.linalg.norm(pts[0] - pts[3]) + 1e-6
    return vert / horiz

def normalized_point(point, w, h):
    return (point.x * w, point.y * h)

def main(video_path="video.mp4", output_csv="eye_frames.csv", sample_fps=1):

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"{video_path} not found")

    cap = cv2.VideoCapture(video_path)
    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30

    sample_every = max(1, int(round(orig_fps / sample_fps)))
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    )

    frame_no = 0
    saved_rows = []

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Frame not read properly")
            break

        frame_no += 1
        if frame_no % sample_every != 0:
            continue
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)
        
        if not results.multi_face_landmarks:
            saved_rows.append({
                "frame": frame_no,
                "eye_openness": np.nan,
                "gaze_x": np.nan,
                "gaze_y": np.nan,
                "blink": 0
            })
            continue

        face_landmarks = results.multi_face_landmarks[0].landmark

        # Eye aspect ratio
        try:
            left_ear = eye_aspect_ratio([(p.x, p.y) for p in face_landmarks], LEFT_EYE_IDX)
            right_ear = eye_aspect_ratio([(p.x, p.y) for p in face_landmarks], RIGHT_EYE_IDX)
            avg_ear = float((left_ear + right_ear)/2.0)
        except Exception:
            avg_ear = np.nan

        # Gaze estimation
        try:
            left_eye_pts = np.array([normalized_point(face_landmarks[i], w, h) for i in LEFT_EYE_IDX])
            left_center = np.mean(left_eye_pts, axis=0)
            left_iris = normalized_point(face_landmarks[LEFT_IRIS_IDX], w, h)
            gaze_x = (left_iris[0] - left_center[0]) / (np.linalg.norm(left_eye_pts[3] - left_eye_pts[0]) + 1e-6)
            gaze_y = (left_iris[1] - left_center[1]) / (np.linalg.norm(left_eye_pts[3] - left_eye_pts[0]) + 1e-6)
        except Exception:
            gaze_x = np.nan
            gaze_y = np.nan

        blink = 1 if (not np.isnan(avg_ear) and avg_ear < 0.20) else 0

        saved_rows.append({
            "frame": frame_no,
            "eye_openness": float(avg_ear) if not np.isnan(avg_ear) else np.nan,
            "gaze_x": float(gaze_x) if not np.isnan(gaze_x) else np.nan,
            "gaze_y": float(gaze_y) if not np.isnan(gaze_y) else np.nan,
            "blink": blink
        })

    cap.release()
    face_mesh.close()

    df = pd.DataFrame(saved_rows)
    df.to_csv(output_csv, index=False)
    print(f"✅ Saved eye tracking CSV -> {output_csv}")

if __name__ == "__main__":

    VIDEO_PATH = "/home/mydeen0302/azure-ml/fraud_detection/videos/interview.mp4"
    OUTPUT_CSV = "/home/mydeen0302/azure-ml/fraud_detection/output/eye_frames.csv"
    FPS = 1

    main(
        video_path=VIDEO_PATH,
        output_csv=OUTPUT_CSV,
        sample_fps=FPS
    )

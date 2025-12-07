# aggregate_features.py
import pandas as pd
import numpy as np
import sys

def aggregate(df, window_frames=10):
    rows = []
    n = len(df)
    # Ensure necessary columns exist
    required_cols = ["frame", "eye_openness", "gaze_x", "gaze_y", "blink"]
    for col in required_cols:
        if col not in df.columns:
            df[col] = np.nan
    df = df[required_cols]

    # Drop rows with NaN frames
    df = df.dropna(subset=["frame"])
    df["frame"] = df["frame"].astype(int)

    for start in range(0, len(df), window_frames):
        seg = df.iloc[start:start+window_frames]
        if seg.empty:
            continue
        features = {
            "start_frame": int(seg["frame"].iloc[0]),
            "end_frame": int(seg["frame"].iloc[-1]),
            "mean_eye_openness": float(seg["eye_openness"].mean(skipna=True)),
            "std_eye_openness": float(seg["eye_openness"].std(skipna=True) or 0.0),
            "blink_count": int(seg["blink"].sum()),
            "gaze_x_mean": float(seg["gaze_x"].mean(skipna=True)),
            "gaze_x_std": float(seg["gaze_x"].std(skipna=True) or 0.0),
            "gaze_y_mean": float(seg["gaze_y"].mean(skipna=True)),
            "nan_ratio": float(seg.isna().mean().mean())
        }
        rows.append(features)

    return pd.DataFrame(rows)

if __name__ == "__main__":

    INPUT_CSV = "/home/mydeen0302/azure-ml/fraud_detection/output/eye_frames.csv"
    OUTPUT_CSV = "/home/mydeen0302/azure-ml/fraud_detection/output/feature_segments.csv"
    WINDOW_SIZE = 5   # change to 5, 15, etc if needed

    print(f"\n📥 Reading CSV: {INPUT_CSV}")

    try:
        df = pd.read_csv(INPUT_CSV)
    except Exception as e:
        print(f"\n❌ Failed to read CSV: {e}")
        exit(1)

    if df.empty:
        print("\n❌ CSV file is empty. No frames to aggregate.")
        exit(1)

    print("✅ CSV loaded successfully")
    print("📊 Aggregating frames into segments...")

    seg_df = aggregate(df, window_frames=WINDOW_SIZE)

    seg_df.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Saved segments -> {OUTPUT_CSV}")



# aggregate_features.py
import pandas as pd
import numpy as np
import sys
def aggregate(df, window_frames=10):  # Reduced from 10 to 3
    rows = []
    required_cols = ["frame", "eye_openness", "gaze_x", "gaze_y", "blink"]
    for col in required_cols:
        if col not in df.columns:
            df[col] = np.nan
    df = df[required_cols].dropna(subset=["frame"])
    df["frame"] = df["frame"].astype(int)
    
    for start in range(0, len(df), window_frames):
        seg = df.iloc[start:start+window_frames]
        if len(seg) < 2: continue  # Need min 2 frames for derivatives
        
        # ORIGINAL features
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
        
        # NEW: Dynamic/discriminative features
        gaze_x = seg["gaze_x"].fillna(0).values
        gaze_y = seg["gaze_y"].fillna(0).values
        blinks = seg["blink"].fillna(0).values
        
        # Gaze volatility (saccade-like movement)
        features["gaze_volatility"] = float(np.std(np.diff(gaze_x)) + np.std(np.diff(gaze_y)))
        
        # Off-screen dwell time (% time |gaze_x| > 0.3 or |gaze_y| > 0.3)
        features["offscreen_ratio"] = float(np.mean(np.abs(gaze_x) > 0.3) + np.mean(np.abs(gaze_y) > 0.3)) / 2
        
        # Max consecutive no-blink frames
        no_blink_runs = []
        current_run = 0
        for b in blinks:
            if b == 0:
                current_run += 1
                no_blink_runs.append(current_run)
            else:
                current_run = 0
        features["max_no_blink_run"] = float(max(no_blink_runs) if no_blink_runs else 0)
        
        # Left/Right eye asymmetry (you'll need to extract both eyes separately in process_video.py)
        features["eye_asymmetry"] = 0.0  # TODO: implement in process_video.py
        
        rows.append(features)
    
    return pd.DataFrame(rows)

if __name__ == "__main__":

    INPUT_CSV = "/home/mydeen0302/azure-ml/fraud_detection/output/eye_frames.csv"
    OUTPUT_CSV = "/home/mydeen0302/azure-ml/fraud_detection/output/feature_segments.csv"
    WINDOW_SIZE = 10  # change to 5, 15, etc if needed

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
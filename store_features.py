# store_features.py
import chromadb
import pandas as pd
import numpy as np
import os
import sys
import logging

# ---------------- LOGGING CONFIG ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

COLLECTION_NAME = "eye_behavior"
CHROMA_PATH = "/home/mydeen0302/azure-ml/fraud_detection/chroma_db"


def main(features_csv):
    try:
        logging.info(f"📥 Reading CSV file: {features_csv}")
        df = pd.read_csv(features_csv)

    except Exception as e:
        logging.exception(f"❌ Failed to read CSV file: {e}")
        sys.exit(1)

    if df.empty:
        logging.warning("⚠️ CSV is empty. No features to store.")
        return

    try:
        logging.info(f"📦 Connecting to ChromaDB at: {CHROMA_PATH}")
        client = chromadb.PersistentClient(path=CHROMA_PATH)

        collection = client.get_or_create_collection(name=COLLECTION_NAME)
        logging.info(f"✅ Using collection: {COLLECTION_NAME}")

    except Exception as e:
        logging.exception(f"❌ Failed to connect to ChromaDB: {e}")
        sys.exit(1)

    ids, embeddings, metadatas = [], [], []

    for idx, r in df.iterrows():
        try:
            vector = [
                float(np.nan_to_num(r["mean_eye_openness"], nan=0.0)),
                float(np.nan_to_num(r["std_eye_openness"], nan=0.0)),
                float(np.nan_to_num(r["blink_count"], nan=0.0)),
                float(np.nan_to_num(r["gaze_x_mean"], nan=0.0)),
                float(np.nan_to_num(r["gaze_x_std"], nan=0.0)),
                float(np.nan_to_num(r["gaze_y_mean"], nan=0.0)),
                float(np.nan_to_num(r["nan_ratio"], nan=0.0))
            ]

            id_ = f"{int(r['start_frame'])}_{int(r['end_frame'])}"

            metadata = {
                "start_frame": int(r["start_frame"]),
                "end_frame": int(r["end_frame"]),
                "blink_count": int(r["blink_count"])
            }

            ids.append(id_)
            embeddings.append(vector)
            metadatas.append(metadata)

        except Exception as e:
            logging.warning(f"⚠️ Skipped row {idx} due to error: {e}")

    if ids:
        try:
            logging.info(f"📤 Storing {len(ids)} feature vectors into ChromaDB...")
            collection.add(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas
            )

            logging.info(
                f"✅ Successfully stored {len(ids)} vectors into collection '{COLLECTION_NAME}'"
            )

        except Exception as e:
            logging.exception(f"❌ Failed to store vectors in ChromaDB: {e}")
            sys.exit(1)
    else:
        logging.warning("⚠️ No valid records found to save.")


if __name__ == "__main__":

    FEATURES_CSV = "/home/mydeen0302/azure-ml/fraud_detection/output/feature_segments.csv"

    logging.info(f"🔍 Checking for feature file: {FEATURES_CSV}")

    if not os.path.exists(FEATURES_CSV):
        logging.error(f"❌ File not found: {FEATURES_CSV}")
        sys.exit(1)

    main(features_csv=FEATURES_CSV)

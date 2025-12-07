import streamlit as st
import requests
import time
import os

# ================== AIRFLOW CONFIG ==================
AIRFLOW_URL = "http://localhost:8081/api/v1/dags/eye_feature_pipeline/dagRuns"

# Replace with your actual Airflow credentials
AIRFLOW_USERNAME = "admin"
AIRFLOW_PASSWORD = "Mydeen0302@"  # <-- put your real password here

AIRFLOW_AUTH = (AIRFLOW_USERNAME, AIRFLOW_PASSWORD)

# ================== OUTPUT FILE ==================
OUTPUT_FILE = "/home/mydeen0302/azure-ml/fraud_detection/output/fraud_report.txt"

# ================== STREAMLIT UI ==================
st.title("🎥 Fraud Detection System")

uploaded_file = st.file_uploader("Upload Interview Video", type=["mp4", "avi", "mov"])

if uploaded_file:
    video_path = f"/home/mydeen0302/azure-ml/fraud_detection/videos/{uploaded_file.name}"

    # Save uploaded video
    with open(video_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success(f"✅ Video saved at: {video_path}")

    if st.button("🚀 Run Fraud Detection"):
        body = {
            "conf": {
                "video_path": video_path
            }
        }

        # Trigger Airflow DAG
        response = requests.post(AIRFLOW_URL, auth=AIRFLOW_AUTH, json=body)

        if response.status_code == 200:
            st.success("✅ Airflow Pipeline Triggered Successfully")
        else:
            st.error("❌ Failed to trigger Airflow DAG")
            st.write(response.text)

        st.info("⏳ Waiting for result...")

        start_time = time.time()
        timeout = 300  # 5 minutes

        while time.time() - start_time < timeout:
            if os.path.exists(OUTPUT_FILE):
                break
            time.sleep(5)

        if os.path.exists(OUTPUT_FILE):
            st.success("✅ Analysis Completed")

            with open(OUTPUT_FILE, "r") as f:
                report = f.read()

            st.subheader("📄 Fraud Detection Report")
            st.text(report)
        else:
            st.error("❌ Result not found. Check Airflow logs.")

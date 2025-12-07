import os
import json
import shutil
import re
import logging
import chromadb
from openai import AzureOpenAI

# ================= CONFIG =================
CHROMA_PATH = "/home/mydeen0302/azure-ml/fraud_detection/chroma_db"
COLLECTION_NAME = "eye_behavior"

AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")

# Make Airflow-safe output directory (relative to DAG or current working directory)
OUTPUT_DIR = "/home/mydeen0302/azure-ml/fraud_detection/output"

# ================= LOGGING =================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# ================= SAFETY CHECK =================
def check_env():
    missing = []
    if not AZURE_OPENAI_API_KEY: missing.append("AZURE_OPENAI_API_KEY")
    if not AZURE_OPENAI_ENDPOINT: missing.append("AZURE_OPENAI_ENDPOINT")
    if not AZURE_OPENAI_DEPLOYMENT: missing.append("AZURE_OPENAI_DEPLOYMENT")

    if missing:
        logging.error(f"Missing environment variables: {', '.join(missing)}")
        exit(1)
    logging.info("✅ Environment variables are set correctly")

# ================= FETCH FROM CHROMA =================
def fetch_all_segments():
    logging.info("📥 Connecting to ChromaDB...")
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection(COLLECTION_NAME)
    collections = client.list_collections()
    logging.info(f"📚 Available collections: {[c.name for c in collections]}")

    results = collection.get(include=["embeddings", "metadatas"])
    segments = []
    for emb, meta in zip(results["embeddings"], results["metadatas"]):
        segments.append({
            "start_frame": meta["start_frame"],
            "end_frame": meta["end_frame"],
            "blink_count": meta["blink_count"],
            "vector": emb
        })
    logging.info(f"✅ Retrieved {len(segments)} segments from ChromaDB")
    return segments

# ================= FORMAT FOR LLM =================
def format_segments_for_llm(segments):
    text = ""
    for s in segments:
        v = s["vector"]
        text += f"""
Segment {s['start_frame']} - {s['end_frame']}
Mean Eye Openness: {v[0]}
Std Eye Openness: {v[1]}
Blink Count: {s['blink_count']}
Gaze X Mean: {v[3]}
Gaze X Std: {v[4]}
Gaze Y Mean: {v[5]}
NaN Ratio: {v[6]}
"""
    return text

def build_prompt(segments):
    return f"""
You are an advanced AI proctoring and behavioral analysis system.

These are eye-tracking + gaze embeddings captured from a candidate's online interview session.
Each segment is a short window of the interview.

Your task:
1. Analyze eye movement patterns
2. Detect unusual or suspicious behavior
3. Decide likelihood of external help / cheating / impersonation
4. Give FINAL Fraud Score (0 - 100)
5. List most suspicious frame segments
6. Provide clear reasoning

Segments data:
{format_segments_for_llm(segments)}

Respond ONLY in valid JSON format:

{{
  "fraud_score": <integer 0-100>,
  "most_suspicious_segments": ["start-end", "..."],
  "reason": "detailed explanation"
}}
"""

# ================= CALL AZURE OPENAI =================
def call_azure_openai(prompt):
    logging.info("🧠 Sending request to Azure OpenAI...")
    client = AzureOpenAI(
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VERSION,
        azure_endpoint=AZURE_OPENAI_ENDPOINT
    )
    try:
        response = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT,
            messages=[
                {"role": "system", "content": "You are an expert fraud detection and proctoring AI."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=600
        )
        logging.info("✅ Received response from Azure OpenAI")
        return response.choices[0].message.content
    except Exception as e:
        logging.error(f"❌ Error calling Azure OpenAI: {e}")
        raise

# ================= JSON CLEANER =================
def clean_llm_json(text):
    text = text.strip()
    text = re.sub(r"^```json", "", text)
    text = re.sub(r"^```", "", text)
    text = re.sub(r"```$", "", text)
    return text.strip()

# ================= DELETE DB =================
def delete_chroma_db():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
        logging.info("🧹 ChromaDB folder deleted successfully")
    else:
        logging.warning("⚠️ No ChromaDB folder found to delete")

# ================= HUMAN OUTPUT =================
def print_human_readable(result):
    logging.info("📄 Printing human-readable report...")
    if "fraud_score" not in result:
        logging.error("❌ Error while analyzing LLM output")
        logging.info(f"Raw Output: {result.get('raw_output')}")
        return

    score = result["fraud_score"]
    segments = ", ".join(result["most_suspicious_segments"])
    reason = result["reason"]

    logging.info(f"Fraud Score: {score} / 100")
    logging.info(f"Suspicious Segments: {segments}")
    logging.info(f"Reasoning:\n{reason}")

    if score >= 70:
        logging.warning("🚨 Status: HIGH RISK")
    elif score >= 40:
        logging.warning("⚠️ Status: MEDIUM RISK")
    else:
        logging.info("✅ Status: LOW RISK")

# ================= SAVE TO FILE =================
def save_to_text_file(result, raw_response=None):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filepath = os.path.join(OUTPUT_DIR, "fraud_report.txt")

    try:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("=========== FRAUD DETECTION REPORT ===========\n\n")
            if "fraud_score" in result:
                score = result["fraud_score"]
                segments = ", ".join(result["most_suspicious_segments"])
                reason = result["reason"]

                f.write(f"Fraud Score : {score} / 100\n")
                f.write(f"Suspicious Segments : {segments}\n\n")
                f.write("Reasoning:\n")
                f.write(reason + "\n\n")

                if score >= 70:
                    f.write("Status: HIGH RISK\n")
                elif score >= 40:
                    f.write("Status: MEDIUM RISK\n")
                else:
                    f.write("Status: LOW RISK\n")
            else:
                f.write("❌ Error while analyzing LLM output\n")
                f.write("Raw Output:\n")
                f.write(raw_response or result.get("raw_output", "N/A"))
        logging.info(f"💾 Saved output to: {filepath}")
    except Exception as e:
        logging.error(f"❌ Failed to save output file: {e}")

# ================= MAIN =================
def main():
    try:
        check_env()

        segments = fetch_all_segments()
        if not segments:
            logging.warning("❌ No embeddings found")
            return

        prompt = build_prompt(segments)
        raw_response = call_azure_openai(prompt)

        try:
            cleaned = clean_llm_json(raw_response)
            result = json.loads(cleaned)
        except Exception:
            logging.error("❌ Invalid JSON from LLM")
            result = {"error": "Invalid JSON from LLM", "raw_output": raw_response}

        print_human_readable(result)
        save_to_text_file(result, raw_response=raw_response)
        delete_chroma_db()

    except Exception as e:
        logging.exception(f"❌ Pipeline failed: {e}")

if __name__ == "__main__":
    main()

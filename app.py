import os
import tempfile
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from openai import OpenAI

# === Load environment variables ===
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

app = Flask(__name__)
CORS(app)  # Allow frontend communication

# === Prompts used to extract info ===
PROMPT_MAP = {
    "company": "What company is the meeting with?",
    "goal": "What is the goal of the meeting?",
    "location": "Where will the meeting take place?",
    "background": "What is the background of this meeting?"
}

@app.route("/process-audio", methods=["POST"])
def process_audio():
    print(">>> Received POST to /process-audio")

    if 'audio' not in request.files:
        print(">>> ERROR: No audio file found in request")
        return jsonify({"error": "No audio file uploaded"}), 400

    audio_file = request.files['audio']
    print(f">>> Audio file received: {audio_file.filename}")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp:
        temp_path = temp.name
        audio_file.save(temp_path)
        print(f">>> Audio file saved to temp path: {temp_path}")

    # === Step 1: Transcription via Whisper ===
    try:
        print(">>> Starting transcription via OpenAI Whisper API...")
        with open(temp_path, "rb") as audio:
            transcript_response = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio
            )
        transcript = transcript_response.text
        print(">>> Transcription complete.")
        print(">>> TRANSCRIPT:\n" + transcript)
    except Exception as e:
        print(f">>> ERROR during transcription: {e}")
        os.remove(temp_path)
        return jsonify({"error": f"Whisper transcription failed: {str(e)}"}), 500

    os.remove(temp_path)
    print(">>> Temp file deleted")

    # === Step 2: Extract info using GPT ===
    extracted = {}
    for field, question in PROMPT_MAP.items():
        try:
            print(f">>> Asking GPT: {question}")
            gpt_response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You extract specific structured info from a transcript."},
                    {"role": "user", "content": f"TRANSCRIPT:\n{transcript}\n\nQUESTION:\n{question}"}
                ]
            )
            answer = gpt_response.choices[0].message.content.strip()
            extracted[field] = answer
            print(f">>> Extracted [{field}]: {answer}")
        except Exception as e:
            print(f">>> ERROR extracting {field}: {e}")
            extracted[field] = f"[ERROR: {e}]"

    print(">>> Final extracted output:")
    print(extracted)
    return jsonify(extracted)

# === Run test server ===
if __name__ == "__main__":
    print(">>> Starting DEBUG Flask server...")
    app.run(debug=True)

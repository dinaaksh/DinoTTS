from flask import Flask, request, jsonify
from TTS.api import TTS

app = Flask(__name__)
tts = TTS(model_name="tts_models/en/ljspeech/vits", progress_bar=False)

@app.route("/tts", methods=["POST"])
def generate_audio():
    data=request.get_json()
    text=data.get("text", "").strip()
    
    if not text:
        return jsonify({"error": "No text provided"}), 400

    try:
        tts.tts_to_file(text=text, file_path="output.wav")
        return jsonify({"status": "success", "message": "Audio generated"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5005)

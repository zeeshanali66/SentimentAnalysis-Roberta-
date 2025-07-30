from flask import Flask, request, jsonify
from flask_cors import CORS
from gradio_client import Client
import os

app = Flask(__name__)
CORS(app)  # Enable CORS

# Load Gradio Space Client
client = Client("zeeshanali66/MultiLingualSetiment")

@app.route("/")
def home():
    return jsonify({
        "status": "online",
        "message": "Send POST to /predict with JSON: {\"text\": \"your sentence here\"}"
    })

@app.route("/predict", methods=["POST"])
def predict():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    user_input = data.get("text")

    if not user_input:
        return jsonify({"error": "Missing 'text' field in JSON"}), 400

    try:
        result = client.predict(
            text=user_input,
            api_name="/predict"
        )
        
        # If result is like: "Sentiment: Very Positive (93.42% confidence)"
        # We'll extract just the label
        sentiment = result.split(":")[-1].split("(")[0].strip()
        return jsonify({"sentiment": sentiment}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

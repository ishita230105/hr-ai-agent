import os
from flask import Flask, jsonify, request
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
app = Flask(__name__)

# This client setup is the most stable for 2026
client = genai.Client(
    api_key=os.getenv("GOOGLE_API_KEY"),
    http_options=types.HttpOptions(api_version='v1') # Force stable v1
)

# Use the 2026 stable model name
MODEL_ID = "gemini-2.5-flash" 

@app.route('/')
def home():
    return "HR AI Agent: Version 2.5 Stable Active"

@app.route('/api/test-ai', methods=['POST'])
def test_ai():
    try:
        data = request.json
        user_input = data.get("message", "Say hello!")
        
        # New SDK syntax for 2026
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=user_input
        )
        
        return jsonify({"response": response.text})
    except Exception as e:
        # Senior Dev Tip: Detailed logging helps us stop guessing
        print(f"DEBUG ERROR: {e}")
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
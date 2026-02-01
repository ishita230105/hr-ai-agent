import os
import pypdf
import io
import json
import time  # <--- Added time for delays
from flask import Flask, render_template, jsonify, request
from dotenv import load_dotenv
from google import genai
from google.genai import types

# --- 1. PATH SETUP ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', 'templates'))
ENV_PATH = os.path.abspath(os.path.join(BASE_DIR, '..', '.env'))

load_dotenv(dotenv_path=ENV_PATH)

app = Flask(__name__, template_folder=TEMPLATE_DIR)

# --- 2. HELPER FUNCTION ---
def extract_text_from_pdf(pdf_bytes):
    try:
        reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        return ""

# --- 3. AI CLIENT SETUP ---
api_key_val = os.getenv("GOOGLE_API_KEY")

if not api_key_val:
    print(f"❌ ERROR: GOOGLE_API_KEY not found! Looking for .env at: {ENV_PATH}")
    client = None
else:
    # Use v1beta to avoid "responseMimeType" errors if you use config later
    client = genai.Client(
        api_key=api_key_val,
        http_options=types.HttpOptions(api_version='v1beta') 
    )

# Switch to a more stable model if 2.5 keeps failing
MODEL_ID = "gemini-2.5-flash" 

# --- 4. ROUTES ---
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/analyze-cv', methods=['POST'])
def analyze_cv():
    if not client:
        return jsonify({"error": "Server Error: API Key missing."}), 500

    try:
        if 'cv' not in request.files:
            return jsonify({"error": "No files uploaded"}), 400
        
        jd = request.form.get('jd', 'Software Engineer')
        files = request.files.getlist('cv')
        
        results = []

        for file in files:
            if file.filename == '':
                continue
                
            pdf_data = file.read()
            cv_text = extract_text_from_pdf(pdf_data)
            
            if not cv_text.strip():
                continue

            agent_prompt = f"""
            You are an expert HR AI. Analyze this resume against the Job Description.
            
            Job Description: {jd}
            Resume Text: {cv_text}
            
            Return a valid JSON object ONLY (no markdown formatting, no ```json fences).
            The keys must be:
            - "candidate_name"
            - "match_score" (number 0-100)
            - "summary"
            - "key_strengths" (list)
            - "weaknesses" (list)
            """

            # --- RETRY LOGIC START ---
            max_retries = 3
            response_text = ""
            
            for attempt in range(max_retries):
                try:
                    # Attempt the API call
                    response = client.models.generate_content(
                        model=MODEL_ID, 
                        contents=agent_prompt
                    )
                    response_text = response.text
                    break # If successful, exit the retry loop
                except Exception as e:
                    error_msg = str(e)
                    # If it's a 503 (Overloaded), wait and retry
                    if "503" in error_msg and attempt < max_retries - 1:
                        print(f"⚠️ Model overloaded. Retrying in 2 seconds... (Attempt {attempt + 1})")
                        time.sleep(2)
                        continue
                    else:
                        # If it's another error or we ran out of retries
                        raise e 
            # --- RETRY LOGIC END ---

            # Clean Markdown if present
            raw_text = response_text.strip()
            if raw_text.startswith("```json"):
                raw_text = raw_text[7:]
            if raw_text.startswith("```"):
                raw_text = raw_text[3:]
            if raw_text.endswith("```"):
                raw_text = raw_text[:-3]

            try:
                analysis_data = json.loads(raw_text)
                results.append(analysis_data)
            except json.JSONDecodeError:
                results.append({
                    "candidate_name": "Unknown (Parse Error)", 
                    "match_score": 0, 
                    "summary": "AI returned invalid JSON.",
                    "key_strengths": [],
                    "weaknesses": []
                })

        results.sort(key=lambda x: x.get('match_score', 0), reverse=True)
        return jsonify({"status": "Success", "analysis": results})

    except Exception as e:
        print(f"Server Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
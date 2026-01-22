import os
import pypdf
import io
from flask import Flask, render_template, jsonify, request
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load environment variables (Local will use .env, Vercel uses Dashboard settings)
load_dotenv()

# --- 1. UNIVERSAL PATH SETUP ---
# Detects the 'api' folder location and finds 'templates' at the root level
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Moves one level up from 'api/' to the root, then into 'templates/'
TEMPLATE_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', 'templates'))

app = Flask(__name__, template_folder=TEMPLATE_DIR)

# --- 2. HELPER FUNCTION ---
def extract_text_from_pdf(pdf_bytes):
    reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

# --- 3. AI CLIENT SETUP ---
api_key_val = os.getenv("GOOGLE_API_KEY")

# Safety check: Prevent crash if API key is missing
if not api_key_val:
    print("CRITICAL: GOOGLE_API_KEY not found in environment variables.")

client = genai.Client(
    api_key=api_key_val,
    http_options=types.HttpOptions(api_version='v1')
)
MODEL_ID = "gemini-2.5-flash"

# --- 4. ROUTES ---
@app.route('/')
def home():
    # Because we defined template_folder above, just use the filename
    return render_template('index.html')

@app.route('/api/analyze-cv', methods=['POST'])
def analyze_cv():
    try:
        if 'cv' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        
        jd = request.form.get('jd', 'Software Engineer')
        file_storage = request.files['cv']
        
        # Read the file bytes directly
        pdf_data = file_storage.read()
        cv_text = extract_text_from_pdf(pdf_data)
        
        if not cv_text.strip():
            return jsonify({"error": "PDF is empty or unreadable"}), 400
        
        agent_prompt = f"Analyze this Resume against the JD: {jd}\n\nResume: {cv_text}"
        response = client.models.generate_content(model=MODEL_ID, contents=agent_prompt)
        
        return jsonify({"status": "Success", "analysis": response.text})
    except Exception as e:
        return jsonify({"error": str(e)})

# Local development runner
if __name__ == "__main__":
    app.run(debug=True)
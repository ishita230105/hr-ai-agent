import os
from flask import Flask, render_template, jsonify, request
from dotenv import load_dotenv
from google import genai
from google.genai import types
import pypdf
import io

def extract_text_from_pdf(pdf_bytes):
    # This reads the PDF from memory (fast and serverless-friendly)
    reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

load_dotenv()

# 1. Setup absolute paths for templates
api_dir = os.path.dirname(os.path.abspath(__file__))
template_dir = os.path.join(api_dir, '..', 'templates')

# 2. Initialize Flask (ONLY ONE LINE NEEDED)
app = Flask(__name__, template_folder=template_dir)

# 3. Gemini Client Setup
client = genai.Client(
    api_key=os.getenv("GOOGLE_API_KEY"),
    http_options=types.HttpOptions(api_version='v1')
)

MODEL_ID = "gemini-2.5-flash" 

@app.route('/')
def home():
    # Flask now knows to look in template_dir, so just use the filename
    return render_template('index.html')

@app.route('/api/analyze-cv', methods=['POST'])
def analyze_cv():
    try:
        if 'cv' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        
        jd = request.form.get('jd', 'Software Engineer')
        file = request.files['cv']
        
        # 1. Extract Text (Already tested and working!)
        pdf_content = file.read()
        cv_text = extract_text_from_pdf(pdf_content)
        
        # 2. Agentic Prompt: This is the "Reasoning" part
        # We tell the AI to follow a specific thought process
        agent_prompt = f"""
        Role: Senior HR Technical Recruiter
        Task: Autonomously analyze the Resume against the Job Description (JD).
        
        Reasoning Process:
        1. Identify the 'Must-Have' skills in the JD.
        2. Scan the Resume for direct evidence of these skills.
        3. Look for 'Hidden Patterns' (e.g., if they know FastAPI, they likely understand REST APIs).
        4. Determine the 'Cultural Fit' or 'Growth Potential' based on project descriptions.
        
        JD: {jd}
        RESUME: {cv_text}
        
        Output format:
        - Match Score: (0-100)
        - Reasoning: (3-4 sentences explaining your logic)
        - Pros: (List 3 bullet points)
        - Gap Analysis: (What is missing?)
        - Final Decision: (Shortlist / Hold / Reject)
        """
        
        # 3. Call the Brain
        response = client.models.generate_content(
            model=MODEL_ID, 
            contents=agent_prompt
        )
        
        # 4. Return the intelligent analysis
        return jsonify({
            "status": "Success",
            "analysis": response.text
        })

    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/api/test-ai', methods=['POST'])
def test_ai():
    try:
        data = request.json
        user_input = data.get("message", "Say hello!")
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=user_input
        )
        return jsonify({"response": response.text})
    except Exception as e:
        print(f"DEBUG ERROR: {e}")
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
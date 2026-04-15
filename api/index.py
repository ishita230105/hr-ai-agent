import os
import pypdf
import io
import json
import textwrap
import base64
from io import BytesIO
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flask import Flask, render_template, jsonify, request
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

# --- APP SETUP ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', 'templates'))
app = Flask(__name__, template_folder=TEMPLATE_DIR)

# --- AI CONFIG ---
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
MODEL_ID = "gemini-2.5-flash"

ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "candidate_summary": {"type": "string"},
        "match_score": {"type": "integer"},
        "technical_alignment": {"type": "array", "items": {"type": "string"}},
        "missing_critical_skills": {"type": "array", "items": {"type": "string"}},
        "interview_questions": {"type": "array", "items": {"type": "string"}},
        "verdict": {"type": "string", "enum": ["Shortlist", "Consider", "Reject"]}
    },
    "required": ["match_score", "technical_alignment", "verdict"]
}

# --- HELPERS ---
def generate_skill_chart(tech_alignment, missing_skills):
    # Slightly wider canvas to accommodate text
    plt.figure(figsize=(9, 5), facecolor='#f8fafc') 
    
    raw_labels = tech_alignment[:5] + missing_skills[:5]
    
    # MAGIC FIX 1: Wrap long sentences into multiple lines (breaks at 30 characters)
    labels = [textwrap.fill(label, width=30) for label in raw_labels]
    
    values = [10] * len(tech_alignment[:5]) + [3] * len(missing_skills[:5])
    colors = ['#6366f1'] * len(tech_alignment[:5]) + ['#ef4444'] * len(missing_skills[:5])

    plt.barh(labels, values, color=colors)
    plt.title('Resume vs. JD Skill Gap Analysis', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    # MAGIC FIX 2: Explicitly force the left margin to be wider
    plt.subplots_adjust(left=0.35) 

    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    plt.close()
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def extract_text(pdf_bytes):
    reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
    return "".join([page.extract_text() for page in reader.pages])

# --- ROUTES ---
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/analyze-cv', methods=['POST'])
def analyze_cv():
    try:
        jd = request.form.get('jd', 'Software Engineer')
        cv_file = request.files['cv']
        cv_text = extract_text(cv_file.read())

        # The Agentic Prompt
        sys_instr = """
                    You are a FAANG recruiter. Evaluate the resume against the JD using strictly structured JSON.
                    CRITICAL: For 'technical_alignment' and 'missing_critical_skills', return ONLY short 1-3 word keywords (e.g., 'Python', 'AWS', 'System Design'). Do NOT return full sentences.
                    """
        user_prompt = f"JD: {jd}\n\nResume: {cv_text}"

        response = client.models.generate_content(
            model=MODEL_ID,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=sys_instr,
                response_mime_type="application/json",
                response_schema=ANALYSIS_SCHEMA
            )
        )
        
        analysis_data = json.loads(response.text)
        chart_b64 = generate_skill_chart(
            analysis_data.get('technical_alignment', []),
            analysis_data.get('missing_critical_skills', [])
        )
        
        return jsonify({
            "status": "Success", 
            "data": analysis_data,
            "chart": chart_b64
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
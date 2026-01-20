import os
from flask import Flask, jsonify, request
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables (API Keys)
load_dotenv()

app = Flask(__name__)

# Initialize our LLM (The Brain)
# We'll use GPT-4o-mini because it's fast and cheap for development
llm = ChatOpenAI(model="gpt-4o-mini")


@app.route('/')
def home():
    return "HR AI Agent API is running!"


@app.route('/api/test-ai', methods=['POST'])
def test_ai():
    # A simple agentic thought process test
    data = request.json
    user_input = data.get("message", "Say hello!")

    prompt = ChatPromptTemplate.from_template("You are an HR expert. Answer this: {input}")
    chain = prompt | llm

    response = chain.invoke({"input": user_input})
    return jsonify({"response": response.content})


if __name__ == "__main__":
    app.run(debug=True)
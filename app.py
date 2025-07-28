from flask import Flask, render_template, jsonify, request
import os
import fitz  # PyMuPDF
from flask_cors import CORS
from together import Together # Import the official Together library
from dotenv import load_dotenv
import logging

# --- Configuration ---
# Set up basic logging
logging.basicConfig(level=logging.INFO)

# Load environment variables from .env file
load_dotenv()

# Configure folders
TEMPLATE_FOLDER = os.path.join(os.path.dirname(__file__), "templates")
# The PDF document to be queried must be placed in this folder
DATA_FOLDER = os.path.join(os.path.dirname(__file__), "data")
PDF_FILENAME = "document.pdf"  # The name of your PDF file

# Initialize Flask app
app = Flask(__name__, template_folder=TEMPLATE_FOLDER)
CORS(app)

# --- AI Model and API Configuration ---
# The API token is now the only required environment variable for the client
API_TOKEN = os.getenv("API_TOKEN")

# --- Global Variable for PDF Text ---
# This will hold the extracted text from the PDF to avoid re-reading the file on every request
PDF_TEXT = ""

# --- Helper Functions ---
def extract_pdf_text(pdf_path):
    """
    Extracts all text from a given PDF file.
    Logs an error if the file is not found or cannot be processed.
    """
    global PDF_TEXT
    if not os.path.exists(pdf_path):
        logging.error(f"PDF file not found at: {pdf_path}")
        PDF_TEXT = "Error: Document not found. Please ensure the PDF file exists in the 'data' directory."
        return

    try:
        with fitz.open(pdf_path) as doc:
            # Concatenate text from all pages
            full_text = "".join(page.get_text() for page in doc)
            PDF_TEXT = full_text
            logging.info(f"Successfully loaded and processed '{PDF_FILENAME}'.")
    except Exception as e:
        logging.error(f"Failed to read or process PDF file: {e}")
        PDF_TEXT = f"Error: Failed to process the PDF document. {e}"


# --- Flask Routes ---
@app.route('/')
def index():
    """
    Serves the main chatbot interface page.
    """
    return render_template('index.html')

@app.route('/api/ask', methods=['POST'])
def ask_question():
    """
    Receives a question from the user, queries the AI model with the PDF context,
    and returns the AI's answer using the Together Python library.
    """
    # Ensure the PDF text has been loaded
    if "Error:" in PDF_TEXT:
        return jsonify({"answer": PDF_TEXT}), 500

    # Get the user's question from the request body
    data = request.json
    question = data.get('question')

    if not question:
        return jsonify({"error": "No question provided."}), 400

    # Construct the prompt for the AI model
    prompt = ("You are Rupini Raman. Your personality is professional and helpful. "
        "Answer the following question based *only* on the information contained in your resume text provided below. "
        "You must speak in the first person (e.g., 'I worked at...', 'My skills include...'). "
        "If the answer is not in your resume, politely state that the information is not available in your resume, for example: "
        "'That information is not something I've included in my resume.'\n\n"
        "--- MY RESUME TEXT ---\n"
        f"{PDF_TEXT}\n\n"
        "--- QUESTION ---\n"
        f"{question}"
    )

    try:
        # Initialize the Together client
        client = Together(api_key=API_TOKEN)

        # API Call to AI Model from together.ai website
        response = client.chat.completions.create(
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.5,
        )
        
        answer = response.choices[0].message.content.strip()
        return jsonify({"answer": answer})

    except Exception as e:
        logging.error(f"Error connecting to Together.ai service: {e}")
        return jsonify({"answer": f"An error occurred while contacting the AI service: {e}"}), 500

# --- Application Startup ---
if __name__ == '__main__':
    # Load the PDF text into memory when the application starts
    pdf_full_path = os.path.join(DATA_FOLDER, PDF_FILENAME)
    extract_pdf_text(pdf_full_path)
    
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000, debug=False)

# This line is for serverless deployments (e.g., Vercel)
handler = app

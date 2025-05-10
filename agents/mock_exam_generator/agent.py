from google.adk.agents import Agent
from PyPDF2 import PdfReader
from fpdf import FPDF
import google.generativeai as genai
import os
from datetime import datetime

def generate_mockup():

    pdf_path = "./mock_exam_generator/mocks/es1AM-Classe_I_25-01-15.pdf"

    # 1. Extract text from the PDF
    text = ""
    reader = PdfReader(pdf_path)
    for page in reader.pages:
        text += page.extract_text() or ""  # Handle None case

    # 2. Generate Similar Text using Gemini
    genai.configure(api_key="AIzaSyAc1bkPAQ9pE-EL1llkaqAoqNDjv1uscVI")
    model = genai.GenerativeModel('gemini-1.5-pro')

    response = model.generate_content(
        f"Generate a new document that follows a similar style, tone, and structure to this: {text}",
        tools=None
    )
    generated_text = response.text

    # 3. Ensure the output folder exists
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    # Fix: Replace unsupported Unicode chars with '?'
    safe_text = generated_text.encode('latin-1', errors='replace').decode('latin-1')
    pdf.multi_cell(0, 10, safe_text)

    # 5. Save to the target folder
    output_folder = "./mock_exam_generator/mocks/generated_mocks"

    # Generate a filename like: "mock_20240510_142356.pdf"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_folder, f"mock_{timestamp}.pdf")
    pdf.output(output_path)


root_agent = Agent(
    model='gemini-2.0-flash-live-001',
    name='root_agent',
    description='Generates mock exam questions or exercises based on past exams',
    instruction='''Generate a mock exam based on the provided image files using the generate_mockup tool. 
                The exam should be similar in style and content to the text of the original document, but with different questions and exercises.
                You don't need the file to be uploaded by the user nor to ask the file path, you already know that the path is "./mock_exam_generator/mocks/es1AM-Classe_I_25-01-15.pdf".
                The output should be a new PDF file containing the generated exam questions and exercises.
                ''',
    tools=[generate_mockup]
)
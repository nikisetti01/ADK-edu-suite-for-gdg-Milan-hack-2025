from google.adk.agents import Agent
from PyPDF2 import PdfReader
from fpdf import FPDF
import google.generativeai as genai
import os
import requests

# Solution: Use a Unicode-aware font and set the encoding
pdf_path = "./real_mocks/es1AM-Classe_I_25-01-15.pdf"

# 1. Extract text from the PDF
text = ""
reader = PdfReader(pdf_path)
for page in reader.pages:
    text += page.extract_text() or ""  # Handle None case

# 2. Generate Similar Text using Gemini
genai.configure(api_key="AIzaSyAc1bkPAQ9pE-EL1llkaqAoqNDjv1uscVI")  # Add your API key here
model = genai.GenerativeModel('gemini-1.5-pro')

response = model.generate_content(
    f"Generate a new document that follows a similar style, tone, and structure to this: {text}",
    tools=None
)
generated_text = response.text

# 3. Convert generated text to PDF (quick fix for Unicode)
pdf = FPDF()
pdf.add_page()
pdf.set_font("Arial", size=12)

# Replace unsupported Unicode chars with '?'
safe_text = generated_text.encode('latin-1', errors='replace').decode('latin-1')
pdf.multi_cell(0, 10, safe_text)

pdf.output("generated.pdf")
print("PDF generated successfully!")
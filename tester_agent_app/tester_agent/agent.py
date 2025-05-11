from google.adk.agents import Agent
from PyPDF2 import PdfReader
import google.generativeai as genai

exam_path = "./tester_agent/completed_exams/rivoluzione_francese.pdf"
book_path = "./tester_agent/ground_truth/rivoluzione_francese.pdf"

def test_the_user():
    
    # 1. Extract text from the PDF of the completed exam
    exam = ""
    reader = PdfReader(exam_path)
    for page in reader.pages:
        exam += page.extract_text() or ""  # Handle None case

    # 2. Extract text from the PDF of the book
    book = ""
    reader = PdfReader(book_path)
    for page in reader.pages:
        book += page.extract_text() or ""  # Handle None case

    # 3. Compare the two texts and generate a response
    genai.configure(api_key="AIzaSyAc1bkPAQ9pE-EL1llkaqAoqNDjv1uscVI")
    model = genai.GenerativeModel('gemini-1.5-pro')

    response = model.generate_content(
        f"Compare the following two texts. The first text is the exam completed by the user: {exam} and the second text is the book (ground truth): {book}. You must not provide the correct answer to the questions the user answered wrongly, but rather tell him which chapter he should revise based on the mistakes he made.",
        tools=None
    )
    
    return response.text

    


root_agent = Agent(
    model='gemini-2.0-flash-live-001',
    name='root_agent',
    description='An assistant that check the answer given from the user to the questions of an exam and tells him which chapter he should revise based on the mistakes he made.',
    instruction=f'''
    Upon request, call `test_the_user()` to check the user's exam answers against the book. Follow these rules:  
    1. **Wrong Answer Definition**: Mark an answer as wrong *only* if it contains factual inaccuracies (e.g., wrong events, dates, or false claims).  
    2. **Correct Answer Definition**: Accept answers as correct if they:  
        - Match the book's facts, even with different wording.  
        - Omit non-essential details but are factually sound.  
        - Add accurate context not in the book, provided it doesn’t contradict the book.  
    3. **Output**: Return a list *only* of answers with factual errors, each paired with the relevant chapter to revise (e.g., "Answer 4 is wrong. Revise Chapter 2: The Bastille").  
    4. **Do not flag answers** for lacking nuance or extra details if they are factually correct.  
    The exam along with questions and answers is provided in the PDF file located at {exam_path}.
    The book (ground truth) is provided in the PDF file located at {book_path}.
    ''',
    tools=[test_the_user]
)
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
    instruction='''
    Answer to the user by not providing the correct answer to the question,
    but listing only for each wrongly answered question, along with the specific wrong sentence in the answer,
    upon request from the user.
    Consider that each question corresponds to a chapter sequentially.
    After that, tell the user which chapter he should revise based on the mistakes he made using the test_the_user tool.
    An answer is to be considered wrong only if it contains some wrong information.
    It means that even if incomplete, along the content is true the answer is to be considered as correct.    
    Do not request revisions for incomplete answers unless they contain false claims.
    You don't need the user to provide you any information, you already have the questions along with the answer provided by the user in the file located in "./tester_agent/completed_exams/rivoluzione_francese.pdf"
    and the book to be used to check the answers is located in "./tester_agent/ground_truth/rivoluzione_francese.pdf".
    Provide feedback only on wrong answers, while do not mention the correct answers.
    Do not ask any other question or perform any other task.
    ''',
    tools=[test_the_user]
)
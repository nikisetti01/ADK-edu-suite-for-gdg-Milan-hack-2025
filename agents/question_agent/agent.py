from google.adk.agents import Agent
import random
import csv


def get_random_question():
    with open('question_agent/flashcard_riv_franc.csv', 'r') as file:
        questions = [row[0] for row in csv.reader(file)]
    return random.choice(questions)


root_agent = Agent(
    model='gemini-2.0-flash-001',
    name='root_agent',
    description='An agents that retrieves a question from a set of questions',
    instruction='Provide the user a random question picked from a list of questions using the get_question tool',
    tools=[get_random_question]
)
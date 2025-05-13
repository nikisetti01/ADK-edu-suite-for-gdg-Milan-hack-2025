# 🎓 ADK Educational Multi-Agent Suite

A modular backend suite demonstrating how to use agents built with **ADK (Agent Development Kit)** and **tool calling** to create interactive and intelligent educational systems.

Each module shows a real-world use case where multiple agents collaborate via structured communication protocols to support and enhance the student learning process.

---

## 📁 Project Structure

| Folder                   | Description |
|--------------------------|-------------|
| `calendar/`              | Creates a personalized **study calendar** for an exam or goal using a planning agent that organizes tasks and dates. |
| `mock_exam_generator_app/` | An agent that **generates mock exams** from examples provided by the student. |
| `question_agent/`        | Helper agent for question formulation and analysis. |
| `socratic_agent/`        | A multi-agent system that engages the student in a **socratic dialogue loop**, refining responses through targeted questioning and performance analysis. |
| `tester_agent_app/`      | An agent that **grades exams** and provides personalized feedback on what to review. |
| `.env.example`           | Sample environment configuration file. |
| `requirements.txt`       | Python dependencies needed to run the project. |

---

## 🧠 How It Works

Each module consists of:

- A **FastAPI** application (`fastapi_app.py`) exposing agent functionality through a REST API.
- One or more **ADK agents** that collaborate to complete educational tasks (e.g., correction, planning, tutoring).

Agents communicate using a **message-passing protocol**, enabling scalable and coordinated multi-agent workflows tailored to education.

---

## 🚀 Highlighted Examples

### 🧠 `socratic_agent/`
Implements a full tutoring loop:

1. A **question agent** asks a preparatory question.
2. The student responds.
3. An **analysis agent** identifies weak points.
4. A **socratic agent** generates a new question focused on the gaps.
5. A **fitness agent** evaluates the learning progress:
   - If improvement stagnates → the student is redirected to study relevant material.
   - If fitness improves → the student receives a new, unrelated question.

The **fitness agent** also uses a **penalty function** to gamify progress and maintain challenge.

---

### 📅 `calendar/`
Builds a **personalized study schedule** based on:
- Student availability
- Goal deadline (e.g., exam date)
- Current preparation state

---

## ⚙️ Setup Instructions

Clone the repository:

```bash
git clone https://github.com/your-username/adk-edu-agents-example.git
cd adk-edu-agents-example

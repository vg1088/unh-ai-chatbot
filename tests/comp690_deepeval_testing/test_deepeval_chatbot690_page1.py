import pytest
import requests
import os
import configparser
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase

# Load the API key from config.txt and set it as an environment variable
config = configparser.ConfigParser()
config.read("config.txt")

try:
    os.environ["OPENAI_API_KEY"] = config.get("settings", "openai_key")
except (configparser.NoSectionError, configparser.NoOptionError) as e:
    raise ValueError("OpenAI API key not found in config.txt under [settings] section.") from e

# Define the test cases with expected answers for COMP690 Page 1
test_cases = [
    {"question": "What time the class schedule?", "expected_answer": "The M2 section is Wednesday from 9:10 AM to 12:00 PM, and the M3 section is Wednesday from 1:10 PM to 4:00 PM."},
    {"question": "Office hours?", "expected_answer": "Professor Karen Jin's office hours are Monday from 1:00 PM to 4:00 PM and Friday from 9:00 AM to 12:00 PM."},
    {"question": "Where’s the class?", "expected_answer": "Room P142."},
    {"question": "How many credits?", "expected_answer": "4 credits."},
    {"question": "Is there a Zoom link?", "expected_answer": "Yes, the Zoom link for Professor Karen Jin’s office hours is Join our Cloud HD Video Meeting."},
    {"question": "Who's the instructor?", "expected_answer": "Professor Karen Jin, Associate Professor in the Department of Applied Engineering and Sciences."},
    {"question": "When is Karen Jin's office hours?", "expected_answer": "Monday 1:00 PM to 4:00 PM and Friday 9:00 AM to 12:00 PM."},
    {"question": "What time does the M2 section start on Wednesday?", "expected_answer": "9:10 AM."},
    {"question": "How can I schedule an appointment with Professor Jin?", "expected_answer": "Email Professor Jin at karen.jin@unh.edu."},
    {"question": "Where is Professor Jin's office located?", "expected_answer": "Room 139, Pandora Mill building."},
    {"question": "What is the professor's email address?", "expected_answer": "karen.jin@unh.edu"},
    {"question": "What room is the class in?", "expected_answer": "Room P142."},
    {"question": "Can I meet the professor on Monday afternoon?", "expected_answer": "Yes, during office hours from 1:00 PM to 4:00 PM."},
    {"question": "Will Professor Jin be available on Fridays?", "expected_answer": "Yes, from 9:00 AM to 12:00 PM."},
    {"question": "How do I join the professor's Zoom for office hours?", "expected_answer": "Use this link: Join our Cloud HD Video Meeting."},
    {"question": "Can I make an appointment outside of office hours?", "expected_answer": "Yes, email Professor Jin at karen.jin@unh.edu."},
    {"question": "Can I email the professor to set up a meeting?", "expected_answer": "Yes, email her at karen.jin@unh.edu."},
    {"question": "What kind of projects will we do in this internship?", "expected_answer": "Team-based projects involving real-world IT products, processes, or services with external stakeholders."},
    {"question": "Wht is prof jin's email?", "expected_answer": "karen.jin@unh.edu"},
    {"question": "When are ofice hurs for prof Jin?", "expected_answer": "Monday 1:00 PM to 4:00 PM and Friday 9:00 AM to 12:00 PM."}
]

# Helper function to get chatbot response
def get_chatbot_response(session, question):
    try:
        response = session.post("http://localhost:1896/llm_response", data={"message": question})
        response.raise_for_status()
        return response.text.strip()
    except requests.exceptions.RequestException as e:
        print(f"Error connecting to chatbot API: {e}")
        return None

# Explicitly set the context to COMP893
def set_course_context(session):
    context_question = "I am asking about COMP690"
    response = get_chatbot_response(session, context_question)
    if response is None:
        print("Failed to set course context.")

@pytest.mark.parametrize("test_case", test_cases)
def test_chatbot_responses(test_case):
    session = requests.Session()
    set_course_context(session)
    
    question = test_case["question"]
    expected_answer = test_case["expected_answer"]
    
    # Get chatbot response
    actual_response = get_chatbot_response(session, question)
    if actual_response is None:
        pytest.fail(f"Chatbot did not return a response for question: {question}")
    
    # Function to clean chatbot responses
    def clean_response(response):
        return response.split(" If you have any more questions")[0].strip()

    # Clean the responses
    cleaned_actual_response = clean_response(actual_response)
    cleaned_expected_answer = clean_response(expected_answer)
    
    # Initialize the AnswerRelevancyMetric
    relevancy_metric = AnswerRelevancyMetric(
        threshold=0.5,
        model="gpt-4",
        include_reason=True,
        async_mode=False  # Disable asynchronous execution
    )
    
    # Create the test case for relevancy
    relevancy_test_case = LLMTestCase(
        input=question,
        actual_output=cleaned_actual_response
    )
    
    # Evaluate relevancy
    relevancy_metric.measure(relevancy_test_case)
    relevance_score = relevancy_metric.score
    relevance_reason = relevancy_metric.reason
    
    # Log the result
    print(f"Question: {question}")
    print(f"Actual Response: {actual_response}")
    print(f"Cleaned Response: {cleaned_actual_response}")
    print(f"Relevance Score: {relevance_score}")
    print(f"Reason: {relevance_reason}")
    
    # Adjusted threshold
    assert relevance_score >= 0.5, f"Failed relevance for question: {question}, Score: {relevance_score}, Reason: {relevance_reason}"

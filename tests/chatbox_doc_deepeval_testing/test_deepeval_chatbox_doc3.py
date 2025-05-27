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

# Define the test cases with expected answers for chatbox_doc Page 3
test_cases = [
    {"question": "Can I use a previous internship to fulfill the course requirements?", "expected_answer": "No, you must have an internship while enrolled in the course."},
    {"question": "I did an internship last summer; can I use that?", "expected_answer": "No, you must have an active internship during the course."},
    {"question": "How do I register my internship experience on Handshake?", "expected_answer": "Log in, go to 'Experiences,' click 'Request an Experience,' complete the form, get approval from your supervisor and instructor."},
    {"question": "How do I add my internship to Handshake?", "expected_answer": "Follow the steps to 'Request an Experience' on Handshake, ensuring supervisor and instructor approval."},
    {"question": "What information do I need to put in Handshake?", "expected_answer": " In Handshake, you'll need to provide details such as your resume, cover letter, internship preferences, and any relevant work experience. It's also helpful to include your skills, education, and a professional profile picture. This will make it easier for employers to find you and for you to apply to internship opportunities."},
    {"question": "Where do I find the 'Request an Experience' form on Handshake?", "expected_answer": "In the 'Experiences' section of your Handshake account."},
    {"question": "Who needs to approve my internship on Handshake?", "expected_answer": "Your internship on Handshake needs to be approved by the faculty internship coordinator."},
    {"question": "How many learning objectives should I include?", "expected_answer": "At least three well-developed learning objectives."},
    {"question": "Where can I get help with Handshake?", "expected_answer": "You can get help with Handshake by visiting the Career Services office at your institution, or you can check the support section on the Handshake website. They often have resources like guides and FAQs to assist you."},
    {"question": "Do I have to do anything else besides the internship itself?", "expected_answer": "Yes, you must complete class requirements (attend meetings, submit logs, final report, presentations)."},
    {"question": "What are the requirements for this internship class?", "expected_answer": "Attend all class meetings, submit weekly logs, complete a final report, and give progress presentations."},
    {"question": "How many class meetings do I need to attend?", "expected_answer": "All scheduled meetings."},
    {"question": "How often are weekly logs due?", "expected_answer": "Every week you work at your internship."},
    {"question": "Are there any presentations required?", "expected_answer": "Yes, progress presentations during the class."},
    {"question": "What's involved in the final report?", "expected_answer": "The details are in the specific course syllabus."},
    {"question": "What are all the requirements for credit in this course?", "expected_answer": "Attending all class meetings, submitting weekly logs, completing a final internship report, and giving progress presentations."},
    {"question": "Is attending every class meeting mandatory?", "expected_answer": "Attending every class meeting is typically required for the internship courses."},
    {"question": "What is the format for weekly logs?", "expected_answer": "The specific format is in the course syllabus."},
    {"question": "What kind of internship report is required?", "expected_answer": "For the internship report, you typically need to provide a final document that summarizes your internship experience, including the tasks you undertook, skills developed, and lessons learned during your time at the organization."},
    {"question": "How many presentations should I expect to give?", "expected_answer": "The syllabus will have the details; the phrasing implies more than one."}
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
    context_question = "I am asking about general internship questions"
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

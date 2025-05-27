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

# Define the test cases with expected answers for COMP690 Page 6
test_cases = [
    {"question": "What is UNH's policy on academic integrity?", "expected_answer": "The UNH Academic Integrity Policy is available online; the syllabus provides a link to this policy."},
    {"question": "What is the policy regarding mandatory reporting of sexual violence or harassment?", "expected_answer": "Faculty members are required to report incidents of sexual violence or harassment to the university's Title IX Coordinator."},
    {"question": "How can I report sexual violence or harassment confidentially?", "expected_answer": "Contact the SHARPP Center for Interpersonal Violence Awareness, Prevention, and Advocacy."},
    {"question": "What resources are available for confidential support at UNH Manchester?", "expected_answer": "SHARPP Extended Services Coordinator (in person and via Zoom), YWCA NH, and the Mental Health Center of Greater Manchester."},
    {"question": "What are the contact details for the UNH Manchester Title IX Deputy Intake Coordinator?", "expected_answer": "The syllabus lists Lisa Enright's email address (lisa.enright@unh.edu) and phone number."},
    {"question": "What resources are available at the UNH Manchester library?", "expected_answer": "Assistance with research and access to online library resources."},
    {"question": "How can I contact the UNH Manchester library?", "expected_answer": "You can contact the library at 603-641-4173 or unhm.library@unh.edu."},
    {"question": "How can I make a research appointment with a librarian?", "expected_answer": "The syllabus provides a link to instructions on making a research appointment."},
    {"question": "How can I use the library search box?", "expected_answer": "The syllabus provides a link to instructions on using the library search box."},
    {"question": "How can I reserve a study room?", "expected_answer": "The syllabus provides a link to instructions on reserving a study room."},
    {"question": "Where can I find resources for citing sources?", "expected_answer": "The syllabus provides a link to resources for citing sources."},
    {"question": "What is the website for the UNH Manchester Library?", "expected_answer": "The syllabus provides the website for the UNH Manchester Library."},
    {"question": "What resources are available for evaluating sources?", "expected_answer": "The syllabus provides a link to resources for evaluating sources."},
    {"question": "What is the phone number of the SHARPP Center?", "expected_answer": "(603) 862-7233/TTY (800) 735-2964"},
    {"question": "What resources are available to report bias, discrimination, or harassment?", "expected_answer": "Contact the Civil Rights & Equity Office at UNH."},
    {"question": "What are the contact details for the UNH Title IX Coordinator?", "expected_answer": "The syllabus lists Bo Zaryckyj's email address (Bo.Zaryckyj@unh.edu) and phone number."},
    {"question": "What is the name of the app that provides access to reporting options and resources?", "expected_answer": "uSafeUS"},
    {"question": "Where can I find the UNH Academic Integrity Policy?", "expected_answer": "The syllabus includes a link to the UNH Academic Integrity Policy."},
    {"question": "Where can I find more information about what happens when I report an incident?", "expected_answer": "The syllabus mentions a page with information on student reporting options."},
    {"question": "What is the phone number for the 24-hour NH Domestic Violence Hotline?", "expected_answer": "1-866-644-3574"}
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

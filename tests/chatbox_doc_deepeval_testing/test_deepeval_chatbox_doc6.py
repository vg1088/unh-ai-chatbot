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

# Define the test cases with expected outcomes for chatbox_doc Page 6
test_cases = [
    {"question": "What is Note 1 referring to?", "expected_answer": "The note explains that for COMP892 (applied research), the student needs a total of 3 credits which may be taken over multiple semesters and requires a tech job (part-time or full-time)."},
    {"question": "What does Note 2 mean?", "expected_answer": "If COMP890 has already been taken, then only 2 credits total are needed for COMP891 or COMP892; they can be taken over multiple semesters."},
    {"question": "What does Note 3 mean?", "expected_answer": "COMP893 must be finished in one semester and requires 2 credits if COMP890 has already been taken."},
    {"question": "Can a graduate student take COMP890 before completing 9 credits?", "expected_answer": "No, the flowchart shows it requires 9 credits completed first."},
    {"question": "Can an undergraduate student take the internship course before finding an internship?", "expected_answer": "Yes, but only by using the applied research option (COMP690) if they have a tech job."},
    {"question": "Can a graduate student take the internship course before completing the required credits?", "expected_answer": "No, according to the flowchart."},
    {"question": "Are there credit requirements for each course?", "expected_answer": "Yes, the notes on the chart detail credit requirements for each course (COMP890, COMP891, COMP892, COMP893). The specific number of credits depends on prerequisites and whether it is a thesis or project-based program."},
    {"question": "Can any of the courses be taken over multiple semesters?", "expected_answer": "Yes, Notes 1 and 2 on the chart specify that some courses can span multiple semesters."}
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

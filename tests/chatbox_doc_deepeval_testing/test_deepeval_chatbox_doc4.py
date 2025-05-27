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

# Define the test cases with expected answers for chatbox_doc Page 4
test_cases = [
    {"question": "Do I need to write a weekly log every single week?", "expected_answer": "Yes, for every week you work at your internship, until you complete the required hours."},
    {"question": "How often are weekly logs due?", "expected_answer": "Every week you work at your internship."},
    {"question": "How many hours do I need to log for undergraduate credit?", "expected_answer": "For undergraduate credit, you need to complete a minimum of 150 hours of internship work."},
    {"question": "How many hours do I need to log for graduate credit?", "expected_answer": "It's variable, roughly 40 hours per credit hour."},
    {"question": "Do I still need to submit logs after I reach the required hours?", "expected_answer": " Yes, you need to submit weekly logs for the entirety of your internship, even after reaching the required hours. This helps ensure you're documenting your experience consistently."},
    {"question": "What if I don't work for a week? Do I still submit a log?", "expected_answer": "No, you only need to submit logs for the weeks you work."},
    {"question": "Can I start my internship before the class starts?", "expected_answer": "Yes, typically, you can start your internship before the class begins. However, it’s important to ensure that you meet all the requirements outlined in your course syllabus and any internship policies."},
    {"question": "What if my internship starts before the course begins?", "expected_answer": "You can start, but only 20% of your hours will count toward the course credit."},
    {"question": "How many internship hours can I count before the course starts?", "expected_answer": " You can count up to 100 hours of internship work before the course starts, provided these hours are approved by your internship coordinator."},
    {"question": "What should I do if my internship offer arrives after the semester begins?", "expected_answer": " If your internship offer arrives after the semester begins, you should still proceed to accept it, but you may need to check with your academic advisor or the internship coordinator about how it impacts your enrollment or credit requirements."},
    {"question": "I just got an internship offer but the semester already started; what now?", "expected_answer": "Contact Professor Karen Jin; you might be able to late-add to the course or adjust your start date."},
    {"question": "What if my internship start date is after the semester begins?", "expected_answer": "If your internship start date is after the semester begins, you can still enroll in the course, as long as you meet the required hours and complete the necessary assignments. Just be sure to communicate with your internship coordinator about your schedule and any adjustments needed for assignments."},
    {"question": "I'm an F1 student. How do I get CPT authorization for my internship?", "expected_answer": "To get CPT (Curricular Practical Training) authorization for your internship as an F1 student, you typically need to follow these steps:Consult Your Designated School Official, Get an Offer Letter, Complete CPT Application, Receive Updated I-20,  Start Your Internship."},
    {"question": "What documents do I need for CPT authorization?", "expected_answer": "Internship job description or posting, offer letter, and proof of course registration."},
    {"question": "What counts as proof of course registration for CPT?", "expected_answer": "For Curricular Practical Training (CPT), proof of course registration typically includes a copy of your course schedule or an enrollment verification letter from the registrar's office showing that you are officially registered for the internship course."},
    {"question": "What do I need to submit to apply for CPT?", "expected_answer": "Internship job description or posting, offer letter, and proof of course registration."},
    {"question": "I'm an international student. What do I need for CPT?", "expected_answer": "As an international student, to apply for Curricular Practical Training (CPT), you typically need the following: Job Offer,Academic Advisor Approval, CPT Application, Proof of Enrollment."},
    {"question": "Where do I submit the CPT application?", "expected_answer": "You'll need to submit the CPT application to the International Students and Scholars Office (ISSO) at your institution. They typically provide guidance and the necessary forms."},
    {"question": "What if my internship offer letter doesn't have specific dates?", "expected_answer": "If your internship offer letter doesn't specify dates, it's best to reach out to your internship supervisor or contact person to clarify the start and end dates for your internship. Having clear dates is important for fulfilling any credit or Hour requirements for your course."},
    {"question": "How much time is required to process CPT authorization?", "expected_answer": "The document doesn't specify a precise timeframe but indicates a 7-10 day advance submission is needed."}
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

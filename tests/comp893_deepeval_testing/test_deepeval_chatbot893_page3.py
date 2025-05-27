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

# Define the test cases with expected answers for Page 3
test_cases = [
    {"question": "What are the components of the final grade?", "expected_answer": "Class Attendance (10%), Sprint Grade (60%), Homework (10%), and Final Project Report (20%)."},
    {"question": "How much is class attendance worth?", "expected_answer": "10% of the final grade."},
    {"question": "How is the Sprint Grade calculated?", "expected_answer": "Teamwork Grade multiplied by Sprint Grade."},
    {"question": "What determines the Teamwork Grade?", "expected_answer": "Peer evaluation for each of the three sprints; detailed rubrics are to be announced (TBA)."},
    {"question": "What determines the Sprint Grade?", "expected_answer": "The overall Sprint Grade is calculated using the formula: **Sprint Grade = Teamwork Grade * Sprint Grade**. Additionally, class attendance (10%) and homework (30%) also contribute to your final grade."},
    {"question": "What percentage is the final project report worth?", "expected_answer": "The final project report in COMP893 is worth 20% of your overall grade."},
    {"question": "Where can I find the final project report format?", "expected_answer": "You can find the final project report format in Appendix A of the COMP893 syllabus. The report should include several sections, such as a title page, conclusions, and a self-assessment of the project experience."},
    {"question": "What's the policy on late submissions?", "expected_answer": "the policy on late submissions is quite strict. Late submissions may only be granted in exceptional cases such as illness, accidents, or emergencies, provided that these circumstances are properly documented. The student must email the instructor prior to the deadline and explain the situation, along with providing evidence to support their request."},
    {"question": "When might a late submission be considered?", "expected_answer": "Only if the student emails before the deadline, explains the circumstances, and provides evidence."},
    {"question": "What is UNH's attendance policy?", "expected_answer": "Students are responsible for attending scheduled meetings and are expected to abide by the University Policy on Attendance."},
    {"question": "What if I can't make a scheduled meeting?", "expected_answer": "Email the instructor beforehand, explain the situation, and request to be excused. Schedule a meeting to update the instructor."},
    {"question": "How many hours per week should I expect to spend on this course?", "expected_answer": "A minimum of 45 hours per credit per term."},
    {"question": "What should I do if I'm sick and can't come to class?", "expected_answer": "If you're sick and can't make it to class, it's important to email your instructor before the class meeting to explain your circumstances and request to be excused."},
    {"question": "What happens if I don't follow the late submission rules?", "expected_answer": "You may receive no credit for the assignment."},
    {"question": "Are there specific rubrics for teamwork evaluation?", "expected_answer": "teamwork evaluation is based on peer evaluations for each of the three sprints. The detailed rubrics for evaluating teamwork will be announced later. Your overall Sprint Grade will be calculated using the Teamwork Grade along with the Sprint Grade related to the technical aspect of the product and team project management."},
    {"question": "Are there specific rubrics for sprint evaluation?", "expected_answer": "Yes, there are specific rubrics for sprint evaluation, This is based on peer evaluation for each of the three sprints, and You'll receive a team grade for each of the three sprints based on the technical aspects of the product and team project management."},
    {"question": "What's the minimum final project report grade needed to pass the course?", "expected_answer": "you need to earn a minimum of 75% on your final project report grade."},
    {"question": "Is there any homework besides the final report?", "expected_answer": "Yes, additional homework in project management and development tools."},
    {"question": "How much is the additional homework worth?", "expected_answer": "10% of the final grade."},
    {"question": "What's the minimum credit hour workload estimate?", "expected_answer": "45 hours per credit per term."}
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
    context_question = "I am asking about COMP893"
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

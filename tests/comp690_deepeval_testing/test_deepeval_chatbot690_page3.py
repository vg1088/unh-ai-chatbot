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

# Define the test cases with expected answers for COMP690 Page 3
test_cases = [
    {"question": "What are the three components of the final grade?", "expected_answer": "Class Attendance (10%), Sprint Grade (60%), and Final Project Report (20%). There is also Homework (10%)."},
    {"question": "What percentage of the final grade is class attendance?", "expected_answer": "class attendance is worth 10% of your final grade."},
    {"question": "How is the Sprint Grade calculated?", "expected_answer": "It's calculated as: Teamwork Grade x Sprint Grade."},
    {"question": "What is included in the Teamwork Grade?", "expected_answer": "the Teamwork Grade consists of several components, including participation in group work, communication with team members, collaboration, and the overall contribution to the team's success. It's important to actively engage with your peers and contribute effectively to group assignments and projects."},
    {"question": "What is included in the Sprint Grade?", "expected_answer": " the Sprint Grade includes several components like Weekly progress reports that outline your activities and achievements, Engagement and participation in your internship,Deliverables submitted at the end of each sprint, Feedback from your internship supervisor or mentor."},
    {"question": "What is the percentage weight of the final project report?", "expected_answer": "In COMP690, the final project report is weighted at 30% of your overall grade."},
    {"question": "What are the requirements for the final project report?", "expected_answer": " The requirements for the final project report in COMP690 are outlined in the syllabus. Typically, you'll need to include an overview of your internship experience, the skills you developed, and a reflection on how the experience relates to your academic goals."},
    {"question": "What is the policy on late submissions?", "expected_answer": "The policy is very strict and applies only in exceptional cases (illness, accident, emergencies) with proper documentation."},
    {"question": "Under what circumstances might a late submission be accepted?", "expected_answer": "In COMP690, late submissions are typically accepted under specific circumstances such as documented medical emergencies, family emergencies, or other significant unforeseen events."},
    {"question": "What is the university's policy on attendance?", "expected_answer": "Students are responsible for attending scheduled meetings and abiding by the University Policy on Attendance (as stated in the UNH Student Rights, Rules, and Responsibilities)."},
    {"question": "What should students do if they cannot attend a scheduled meeting?", "expected_answer": "Email the instructor BEFORE the meeting to explain the circumstances and request to be excused. Arrange a meeting to update internship progress."},
    {"question": "What is the minimum number of hours required for the course?", "expected_answer": "A minimum of 45 hours of student academic work per credit per term."},
    {"question": "What should students do if they cannot attend class due to illness?", "expected_answer": "Email the instructor before class, explaining the situation and requesting an excused absence. Follow the late submission policy if applicable."},
    {"question": "What happens if a student fails to comply with late submission rules?", "expected_answer": "In COMP690, if a student fails to comply with late submission rules, it typically results in a penalty, which may include a reduction in the submitted work's grade or no credit being given for that work."},
    {"question": "What are the detailed rubrics for evaluating teamwork?", "expected_answer": "The detailed rubrics are TBA (To Be Announced)."},
    {"question": "What are the detailed rubrics for evaluating sprints?", "expected_answer": "The detailed rubrics are TBA (To Be Announced)."},
    {"question": "What is the minimum grade needed on the final report to pass the course?", "expected_answer": "To pass COMP690, you need to achieve at least a grade of C or higher on your final report."},
    {"question": "Is there any additional homework besides the final report?", "expected_answer": "In COMP690, there are primarily two components you need to complete: the final report and any other assignments specified in the syllabus."},
    {"question": "What is the weight of the homework towards the final grade?", "expected_answer": "In COMP690, homework assignments typically account for 30% of your final grade."},
    {"question": "What is the credit hour workload estimate for this course?", "expected_answer": "A minimum of 45 hours per credit per term."}
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

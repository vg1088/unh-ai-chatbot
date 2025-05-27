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

# Define the test cases with expected answers and retrieval context
test_cases = [
    {
        "question": "Student learning outcome for COMP690?",
        "expected_answer": "The student learning outcomes for COMP690 Internship Experience are as follows: 1. Analyze complex computing problems and identify solutions by applying principles of computing. 2. Design, implement, and evaluate computing solutions that meet IT computing requirements. 3. Communicate effectively in a variety of professional contexts. 4. Function effectively as a member or leader of a team engaged in IT activities. 5. Identify and analyze user needs in the process of developing and operating computing systems.",
        "retrieval_context": ["The student learning outcomes for COMP690 Internship Experience are as follows: 1. Analyze complex computing problems and identify solutions by applying principles of computing. 2. Design, implement, and evaluate computing solutions that meet IT computing requirements. 3. Communicate effectively in a variety of professional contexts. 4. Function effectively as a member or leader of a team engaged in IT activities. 5. Identify and analyze user needs in the process of developing and operating computing systems."]
    },
    {
        "question": "How much of the grade is class attendance in COMP893?",
        "expected_answer": "In Fall 2024 semester, 10% of the grade is based on class attendance",
        "retrieval_context": ["The Fall 2024 semester course syllabus states that 10% of the grade is based on class attendance"]
    },
    {
        "question": "What components does the final grade consist of in COMP893?",
        "expected_answer": " The final grade in Fall 2024 semester consists of four components: 10% Class Attendance of all required meetings. 60% Sprint Grade. 10% Homework and 20% Final Project Report",
        "retrieval_context": [" The Fall 2024 semester course syllabus states that the final grade consists of four components: 10% Class Attendance of all required meetings. 60% Sprint Grade. 10% Homework and 20% Final Project Report"]
    },
    {
        "question": "How is sprint grade calculated?",
        "expected_answer": "The sprint grade in Fall 2024 semester is calculated as the Teamwork Grade multiplied by the Sprint Grade. The Teamwork Grade is based on peer evaluation for each of the three sprints, and the Sprint Grade is based on the technical aspect of the product and team project management",
        "retrieval_context": ["The Fall 2024 semester course syllabus states that The sprint grade in Fall 2024 semester is calculated as the Teamwork Grade multiplied by the sprint Grade. The Teamwork Grade is based on peer evaluation for each of the three sprints, and the Sprint Grade is based on the technical aspect of the product and team project management"]
    },
    {
        "question": "What is the Credit Hour Workload Estimate?",
        "expected_answer": " The Credit Hour Workload Estimate for COMP893 and COMP690 is a minimum of 45 hours of student academic work per credit per term",
        "retrieval_context": ["The Credit Hour Workload Estimate for COMP893 and COMP690 is a minimum of 45 hours of student academic work per credit per term"]
    },
    {
        "question": "What are the attendance policies in COMP893 and COMP690?",
        "expected_answer": " The attendance policy for COMP893 and COMP690 states that students are responsible for attending scheduled meetings and are expected to abide by the University Policy on Attendance. If a student cannot attend a scheduled meeting, they must email the instructor about the circumstances and request to be excused BEFORE the class meeting. Additionally, students need to arrange a meeting with the instructor individually to update their internship progress",
        "retrieval_context": [" The attendance policy for COMP893 and COMP690 states that students are responsible for attending scheduled meetings and are expected to abide by the University Policy on Attendance. If a student cannot attend a scheduled meeting, they must email the instructor about the circumstances and request to be excused BEFORE the class meeting. Additionally, students need to arrange a meeting with the instructor individually to update their internship progress"]
    },
    {
        "question": "What do you do if you think you’ll miss a meeting?",
        "expected_answer": " If you anticipate missing a meeting, you should email the instructor about the circumstances and request to be excused for the meeting BEFORE the class meeting. It is important to communicate in advance and provide a valid reason for your absence",
        "retrieval_context": [" The course syllabus of COMP893 and COMP690 states that a student needs to email the instructor about the circumstances and request to be excused for the meeting BEFORE the class meeting. It is important to communicate in advance and provide a valid reason for any missing meetings."]
    },
    {
        "question": "What is the policy on late submissions?",
        "expected_answer": "A late submission may be granted only if you email prior to the deadline and explains and provides evidence for the circumstances that prevent you from meeting the submission requirement",
        "retrieval_context": ["The policy for late submissions in COMP893 and COMP690 is very strict and applies only in exceptional cases of student illness, accidents, or emergencies that are properly documented. A late submission may be granted only if the student emails prior to the deadline and explains and provides evidence for the circumstances that have prevented them from meeting the class requirement. Failing to comply with these rules may result in no credit for the assignment."]
    },
    {
        "question": "Do I still need to take the course if I am currently working?",
        "expected_answer": "Yes, you do. Even if you are currently working or have worked in the past, you will still need to take the Internship Experience course as a degree requirement. However, you don't need to take another internship position if you are currently working in the field. You may use the applied research option of COMP690, or COMP892 to fulfill the internship requirements.",
        "retrieval_context": ["Yes, you do. Even if you are currently working or have worked in the past, you will still need to take the Internship Experience course as a degree requirement. However, you don't need to take another internship position if you are currently working in the field. You may use the applied research option of COMP690 for undergrad students, or COMP892 for graduate students to fulfill the internship requirements."]
    },
    {
        "question": "I did an internship last summer. Can I use that to cover the internship requirements?",
        "expected_answer": "No, a past internship completed cannot be used to fulfill the requirements for the course. The internship position and required hours must be completed while you are registered in the Internship Experience course.",
        "retrieval_context": ["No, a past internship completed cannot be used to fulfill the requirements for the course. The internship position and required hours must be completed while you are registered in the Internship Experience course."]
    }
]

# Helper function to get chatbot response with context
def get_chatbot_response_with_context(session, question, context):
    try:
        # Combine the context and question
        prompt = f"Context: {context}\n\nQuestion: {question}"
        response = session.post("http://localhost:1896/llm_response", data={"message": prompt})
        response.raise_for_status()
        return response.text.strip()
    except requests.exceptions.RequestException as e:
        print(f"Error connecting to chatbot API: {e}")
        return None

# Explicitly set the course context to general internship questions 
def set_course_context(session):
    context_question = "I am asking about general internship questions"
    response = get_chatbot_response_with_context(session, context_question, "")
    if response is None:
        print("Failed to set course context.")

@pytest.mark.parametrize("test_case", test_cases)
def test_chatbot_responses(test_case):
    session = requests.Session()
    set_course_context(session)
    
    question = test_case["question"]
    expected_answer = test_case["expected_answer"]
    retrieval_context = test_case["retrieval_context"]
    
    # Get chatbot response with retrieval context
    actual_response = get_chatbot_response_with_context(session, question, retrieval_context)
    if actual_response is None:
        pytest.fail(f"Chatbot did not return a response for question: {question}")
    
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
        actual_output=actual_response,
        expected_output=expected_answer,
        retrieval_context=retrieval_context
    )
    
    # Evaluate relevancy
    relevancy_metric.measure(relevancy_test_case)
    relevance_score = relevancy_metric.score
    relevance_reason = relevancy_metric.reason
    
    # Log the result
    print(f"Question: {question}")
    print(f"Retrieval Context: {retrieval_context}")
    print(f"Actual Response: {actual_response}")
    print(f"Expected Output: {expected_answer}")
    print(f"Relevance Score: {relevance_score}")
    print(f"Reason: {relevance_reason}")
    
    # Adjusted threshold
    assert relevance_score >= 0.5, f"Failed relevance for question: {question}, Score: {relevance_score}, Reason: {relevance_reason}"

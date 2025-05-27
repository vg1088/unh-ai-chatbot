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
        "question": "When may I take the internship course COMP690? ",
        "expected_answer": "You may take the internship experience course COMP690 any time you have an internship. Additionally, you can take the COMP690 applied research option if you have a part-time or full-time tech job. If you can't find an internship by the last semester of your program, you are allowed to take COMP690 with the group project option.",
        "retrieval_context": ["For undergraduate students, you may take the internship course COMP690 any time you have an internship. Additionally, you can take the COMP690 applied research option if you have a part-time or full-time tech job. If you can't find an internship by the last semester of your program, you are allowed to take COMP690 with the group project option."]
           
    },
    {
        "question": "When may I take COMP890",
        "expected_answer": "You may take COMP890 Internship and Career Planning after you finish your first semester of study.",
        "retrieval_context": ["Graduate students: COMP890: Internship and Career Planning. This is a 1 cr course you need to take after the first semester to help you plan for the internship search process. The course is offered in fall and spring semesters."]
    },
    {
        "question": "When may I take COMP891",
        "expected_answer": "You may take COMP891 Internship Practice when you have found an external internship. The course is offered in all semesters year around.",
        "retrieval_context": ["Graduate students: COMP891: Internship Practice. This is a variable credit 1-3 crs course that you will take when you have an external internship. You will need to register in this course for at least 1 credit to apply for CPT. The course is offered in both fall and spring semesters, as well as during the summer."]
    },
    {
        "question": "When may I take COMP892",
        "expected_answer": " You may take COMP892 Applied Research Internship if you are currently working full time or part time in the tech fields. The course is offered in all semesters year around.",
        "retrieval_context": ["Graduate students: COMP 892: Applied Research Internship This is a variable credit 1-3 crs course for students who are currently working full time or part time in the tech fields. The course is offered in both fall and spring semesters, as well as during the summer."]
    },
    {
        "question": "When may I take COMP893",
        "expected_answer": "You may take COMP893 Team Project Internship in your last semester of study and need to fulfill the internship requirement. The course is offered in fall and spring semesters.",
        "retrieval_context": ["Graduate students: COMP 893: Team Project Internship The course is for students who are in their last semester of study and need to fulfill the internship requirements. The COMP893 Team Project Internship course is designed for students who want to gain practical skills and insights into the field of computing by working on collaborative projects with external stakeholders. The course is offered in fall and spring semesters."]
    },
    {
        "question": "What is the course name of COMP893?",
        "expected_answer": " The course name of COMP893 is Team Project Internship.",
        "retrieval_context": ["The course information section of the Fall 2024 semester course syllabus states that the name of COMP 893 is Team Project Internship."]
    },
    {
        "question": "What is the course name of COMP690?",
        "expected_answer": "The course name of COMP690 is Internship Experience.",
        "retrieval_context": ["The course information section of the Fall 2024 semester course syllabus states that the name of COMP 690 is Internship Experience."]
    },
    {
        "question": "What room is COMP893 in?",
        "expected_answer": "The classroom of COMP893 is in Room P142 in Fall 2024 semester.",
        "retrieval_context": ["In Fall 2024 semester, COMP893 is located in Room P142"]
    },
    {
        "question": "What room is COMP690 in?",
        "expected_answer": "The classroom of COMP690 is in Room P142 in Fall 2024 semester.",
        "retrieval_context": ["In Fall 2024 semester, COMP690 is located in Room P142"]
    },
    {
        "question": "What time is COMP893?",
        "expected_answer": " COMP893 has two sections in Fall 2024: M1 Section meets on Wednesday 9:10am-12pm and M2 Section meets on Wednesday 1:10-4pm.",
        "retrieval_context": ["COMP893 has two sections: M1 Section meets on Wednesday 9:10am-12pm and M2 Section meets on Wednesday 1:10-4pm"]
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

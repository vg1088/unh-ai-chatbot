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
        "question": "what majors are required to take internship courses? ",
        "expected_answer": "Students from all computing majors are required to take internship courses. This includes undergraduate Computer Information Systems Major (CIS), Computer Science (CS) Major, and at the graduate level: M.S. Information Technology Major and M.S. Cybersecurity Engineering Major.",
        "retrieval_context": [
            "Internship courses are mandatory for all computing major students.",
            "This includes undergraduate Computer Information Systems Major (CIS), Computer Science (CS) Major, and at the graduate level: M.S. Information Technology Major and M.S. Cybersecurity Engineering Major."
        ]
    },
    {
        "question": "what should I do to get an internship?",
        "expected_answer": "You can first start your search from UNH Handshake website. You can also apply for jobs directly through the website of the company or organization you are interested in, or through job websites such as LinkedIn, Indeed and Hired. Do attend internship fair on both Manchester and Durham campus, and speak with family, friends, and faculty. Career and Professional Success office can help you with resume writing, interview coaching and other career advice.",
        "retrieval_context": ["Students are encouraged to start their search on Handshake as most of the employers who posted their jobs are looking for UNH students. Students can also apply for jobs through the website of the company/organization they are interested in, or through job websites such as LinkedIn, Indeed and Hired. Do attend internship fair on both Manchester and Durham campus, and speak with family, friends, and faculty. Career and Professional Success office can help you with resume writing, interview coaching and other career advice."]
    },
    {
        "question": "How can I register my internship experience on Handshake?",
        "expected_answer": "To register your internship experience on Handshake, please follow these steps: 1. Login to Handshake. 2. On the top right corner, under Career Center, go to Experiences. 3. Click on Request an Experience and fill out the form. 4. Your internship experience must be approved by your site supervisor and your course instructor. 5. Make sure to include at least three well-developed learning objectives. If you have any questions related to registering your internship experience on Handshake, please contact the Career and Professional Success office.",
        "retrieval_context": ["To register your internship experience on Handshake, please follow these steps: 1. Login to Handshake. 2. On the top right corner, under Career Center, go to Experiences. 3. Click on Request an Experience and fill out the form. 4. Your internship experience must be approved by your site supervisor and your course instructor. 5. Make sure to include at least three well-developed learning objectives. If you have any questions related to registering your internship experience on Handshake, please contact the Career and Professional Success office."]
    },
    {
        "question": "what are the internship course options for undergrads?",
        "expected_answer": " If you are an undergraduate student, you need to take COMP690 Internship Experience. If you are currently working, you should take the course with the applied research option. If you are in your last semester, you may take the course with the team project option.",
        "retrieval_context": ["Internship Courses: Undergraduate students: • COMP690 Internship Experience The course has an applied research option for students who are currently working, and a team project option for students in their last semester of program."]
    },
    {
        "question": "what are the internship course options for graduate students?",
        "expected_answer": "If you are a graduate student, the internship course options include: - COMP890: Internship and Career Planning - COMP891: Internship Practice - COMP892: Applied Research Internship - COMP893: Team Project Internship",
        "retrieval_context": ["For graduate students, the internship course options include: - COMP890: Internship and Career Planning - COMP891: Internship Practice - COMP892: Applied Research Internship - COMP893: Team Project Internship"]
    },
    {
        "question": "Tell me more about COMP690",
        "expected_answer": "COMP690 Internship Experience is for undergraduate students. It has an applied research option for students who are currently working, and a team project option for students in their last semester of program.",
        "retrieval_context": ["Undergraduate students: COMP690 Internship Experience has an applied research option for students who are currently working, and a team project option for students in their last semester of program."]
    },
    {
        "question": "Tell me more about COMP890",
        "expected_answer": "COMP890 Internship and Career Planning is for graduate students. You need to take this 1 cr course after the first semester to help you plan for the internship search process. It is offered in fall and spring semesters.",
        "retrieval_context": ["Graduate students: COMP890: Internship and Career Planning. This is a 1 cr course you need to take after the first semester to help you plan for the internship search process. The course is offered in fall and spring semesters."]
    },
    {
        "question": "Tell me more about COMP891",
        "expected_answer": "COMP891 Internship Practice is for graduate students. This is a variable credit 1-3 crs course that you will take when you have an external internship. You will need to register in this course for at least 1 credit to apply for CPT. The course is oJered in both fall and spring semesters, as well as during the summer.",
        "retrieval_context": ["Graduate students: COMP891: Internship Practice. This is a variable credit 1-3 crs course that you will take when you have an external internship. You will need to register in this course for at least 1 credit to apply for CPT. The course is offered in both fall and spring semesters, as well as during the summer."]
    },
    {
        "question": "Tell me more about COMP892",
        "expected_answer": "COMP 892: Applied Research Internship is for graduate students. This is a variable credit 1-3 crs course for students who are currently working full time or part time in the tech fields. The course is offered in both fall and spring semesters, as well as during the summer.",
        "retrieval_context": ["Graduate students: COMP 892: Applied Research Internship This is a variable credit 1-3 crs course for students who are currently working full time or part time in the tech fields. The course is offered in both fall and spring semesters, as well as during the summer."]
    },
    {
        "question": "Tell me more about COMP893",
        "expected_answer": "COMP 893: Team Project Internship is for graduate students who are in their last semester of study and need to fulfill the internship requirements. You will work with other students on a collaborative project to gain practical skills and insights into the field of computing. The course is offered in fall and spring semesters.",
        "retrieval_context": ["Graduate students: COMP 893: Team Project Internship The course is for students who are in their last semester of study and need to fulfill the internship requirements. The COMP893 Team Project Internship course is designed for students who want to gain practical skills and insights into the field of computing by working on collaborative projects with external stakeholders. The course is offered in fall and spring semesters."]
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

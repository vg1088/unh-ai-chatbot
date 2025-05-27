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
        "question": "What time is COMP690?",
        "expected_answer": "COMP690 has two sections in Fall 2024: M2 Section meets on Wednesday 9:10am-12pm and M2 Section meets on Wednesday 1:10-4pm",
        "retrieval_context": ["COMP690 has two sections: M2 Section meets on Wednesday 9:10am-12pm and M3 Section meets on Wednesday 1:10-4pm"]
    },
    {
        "question": "Who is the instructor of COMP893?",
        "expected_answer": "The instructor of COMP893 Team Project Internship is Professor Karen Jin",
        "retrieval_context": ["Professor Karen Jin is from the Department of Applied Engineering and Sciences. She is the instructor for COMP893 and COMP690. She is also the computing program internship coordinator"]
    },
    {
        "question": "Who is the instructor of COMP690?",
        "expected_answer": "The instructor of COMP690 Internship Experience is Professor Karen Jin",
        "retrieval_context": [" Professor Karen Jin is from the Department of Applied Engineering and Sciences. She is the instructor for COMP893 and COMP690. She is also the computing program internship coordinator"]
    },
    {
        "question": "What is Karen Jin’s role?",
        "expected_answer": "Karen Jin teaches COMP893 and COMP690. She is also the internship coordinator for the computing programs.",
        "retrieval_context": [" Professor Karen Jin is from the Department of Applied Engineering and Sciences. She is the instructor for COMP893 and COMP690. She is also the computing program internship coordinator"]
    },
    {
        "question": "How to contact Karen Jin?",
        "expected_answer": "You can contact her by email Karen.Jin@unh.edu. Her office is located in Rm139, Pandora Mill building.",
        "retrieval_context": ["Prof. Karen Jin’s office is located in Rm139, Pandora Mill building. Her email is Karen.Jin@unh.edu"]
    },
    {
        "question": "What are the instructor's office hours?",
        "expected_answer": " Karen Jin's office hours are Monday 1-4pm and Friday 9-noon. You can also make an appointment with her to meet in person or over zoom.",
        "retrieval_context": ["Karen Jin's office hours are Monday 1-4pm and Friday 9-noon. She is available in person or over Zoom, and appointments can be made via email"]
    },
    {
        "question": "How do you make appointments with Karen Jin?",
        "expected_answer": "You should email her directly and arrange these meetings in advance. Her email is Karen.Jin@unh.edu.",
        "retrieval_context": [" Email directly the instructor or internship coordinator Karen Jin to make an appointment. It's important to arrange these meetings in advance and provide a clear reason for the meeting. She is available in person or over Zoom, and appointments can be made via email. Her email is Karen.Jin@unh.edu"]
    },
    {
        "question": "What is the course description for COMP893?",
        "expected_answer": " The course description for COMP893 Team Project Internship is as follows: The internship course provides experiential learning experience through placement in team projects. This hands-on experience allows students to gain practical skills and insights into the field of computing. By working on a collaborative project with external stakeholders, they will contribute to the development of real-world information technology products, processes, or services, and understand the challenges involved in implementing technology solutions in a professional setting.",
        "retrieval_context": [" The course description of COMP893 is stated as The internship course provides experiential learning experience through placement in team projects. This hands-on experience allows students to gain practical skills and insights into the field of computing. By working on a collaborative project with external stakeholders, they will contribute to the development of real-world information technology products, processes, or services, and understand the challenges involved in implementing technology solutions in a professional setting."]
    },
    {
        "question": "What is the course description for COMP690?",
        "expected_answer": "The course description for COMP690 Internship Experience is as follows: The internship course provides experiential learning experience through placement in team projects. This hands-on experience allows students to gain practical skills and insights into the field of computing. By working on a collaborative project with external stakeholders, they will contribute to the development of real-world information technology products, processes, or services, and understand the challenges involved in implementing technology solutions in a professional setting.",
        "retrieval_context": ["The course description for COMP690 Internship Experience is as follows: The internship course provides experiential learning experience through placement in team projects. This hands-on experience allows students to gain practical skills and insights into the field of computing. By working on a collaborative project with external stakeholders, they will contribute to the development of real-world information technology products, processes, or services, and understand the challenges involved in implementing technology solutions in a professional setting."]
    },
    {
        "question": "Student learning outcome for COMP893?",
        "expected_answer": "The student learning outcomes for COMP893 Team Project Internship are as follows: 1. Analyze complex computing problems and identify solutions by applying principles of computing. 2. Design, implement, and evaluate computing solutions that meet IT computing requirements. 3. Communicate effectively in a variety of professional contexts. 4. Function effectively as a member or leader of a team engaged in IT activities. 5. Identify and analyze user needs in the process of developing and operating computing systems.",
        "retrieval_context": ["The student learning outcomes for COMP893 Team Project Internship are as follows: 1. Analyze complex computing problems and identify solutions by applying principles of computing. 2. Design, implement, and evaluate computing solutions that meet IT computing requirements. 3. Communicate effectively in a variety of professional contexts. 4. Function effectively as a member or leader of a team engaged in IT activities. 5. Identify and analyze user needs in the process of developing and operating computing systems."]
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

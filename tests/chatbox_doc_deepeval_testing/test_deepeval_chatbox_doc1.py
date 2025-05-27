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

# Define the test cases with expected answers for chatbox_doc Page 1
test_cases = [
    {"question": "What computing majors require internships?", "expected_answer": "Generally, majors like Computer Science, Computer Engineering, and Information Technology often require internships to help students gain practical experience. You may also want to refer to your program's syllabus or official guidelines for specific information about internship requirements."},
    {"question": "Is there an internship option for Cybersecurity Engineering students?", "expected_answer": "Yes, Cybersecurity Engineering students can pursue internships as part of their academic requirements."},
    {"question": "Who is the internship coordinator?", "expected_answer": "The internship coordinator is Karen Jin. You can reach her at karen.jin@unh.edu for any internship-related questions or concerns."},
    {"question": "Where is Karen Jin's office located?", "expected_answer": "Room 139, Pandora Mill building."},
    {"question": "What is Karen Jin's email address?", "expected_answer": "Karen.Jin@unh.edu"},
    {"question": "How can I contact the CaPS office?", "expected_answer": "You can contact them via phone at (603) 641-4394 or visit their website: Career and Professional Success"},
    {"question": "What is the CaPS website address?", "expected_answer": "Career and Professional Success"},
    {"question": "What is the phone number for the CaPS office?", "expected_answer": "(603) 641-4394"},
    {"question": "What is the OISS website?", "expected_answer": "International Students & Scholars"},
    {"question": "What is the OISS email address?", "expected_answer": "oiss@unh.edu"},
    {"question": "What undergraduate internship courses are available?", "expected_answer": "COMP690"},
    {"question": "What is COMP690 about?", "expected_answer": "It's an internship experience course with applied research and team project options."},
    {"question": "Does COMP690 offer a team project option?", "expected_answer": "Yes."},
    {"question": "What is COMP890?", "expected_answer": "A graduate-level internship course: Internship and Career Planning."},
    {"question": "When is COMP890 offered?", "expected_answer": "Fall and spring semesters."},
    {"question": "How many credits is COMP890?", "expected_answer": "1 credit."},
    {"question": "What is COMP891?", "expected_answer": "A graduate-level internship course: Internship Practice."},
    {"question": "What are the graduate-level internship courses?", "expected_answer": "The graduate-level internship courses are COMP690 and COMP893."},
    {"question": "What internship classes are there?", "expected_answer": "For undergraduates: COMP690. For graduates: COMP890 and COMP891."},
    {"question": "Tell me about the graduate internship program.", "expected_answer": "Graduate students can choose between COMP890 (Internship and Career Planning) and COMP891 (Internship Practice)."}
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

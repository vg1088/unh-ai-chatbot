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
        "question": "How to register for Internship Courses?",
        "expected_answer": "You need to get permission from the internship course instructor. For more details, students may contact Prof. Karen Jin, the internship coordinator.",
        "retrieval_context": ["Students can register for internship courses by obtaining the instructor's permission. They need to email the faculty internship coordinator or the course instructor to register for the course. For more details, students may contact Prof. Karen Jin the internship coordinator."]
    },
    {
        "question": "What requirements do you need to fulfill to earn the credit?",
        "expected_answer": "To earn credit for the internship course, you need to fulfill the following requirements: - Attend every scheduled class meeting - Submit weekly logs - Complete a final internship report - Give progress presentations during the class Additionally, you must meet the specific course syllabus requirements and complete the necessary hours at your internship based on your enrolled credit hours.",
        "retrieval_context": ["To earn credit for the internship course, you need to fulfill the following requirements: - Attend every scheduled class meeting - Submit weekly logs - Complete a final internship report - Give progress presentations during the class Additionally, you must meet the specific course syllabus requirements and complete the necessary hours at your internship based on your enrolled credit hours."]
    },
    {
        "question": "What are the steps for registering your internship experience on handshake?",
        "expected_answer": "To register your internship experience on Handshake, follow these steps: 1. Login to Handshake. 2. On the top right corner, under Career Center, go to Experiences. 3. Click on Request an Experience and fill out the form. 4. Ensure that your internship experience is approved by both your site supervisor and your course instructor.",
        "retrieval_context": [" To register your internship experience on Handshake, follow these steps: 1. Login to Handshake. 2. On the top right corner, under Career Center, go to Experiences. 3. Click on Request an Experience and fill out the form. 4. Ensure that your internship experience is approved by both your site supervisor and your course instructor."]
    },
    {
        "question": "Do I need to write weekly logs every week?",
        "expected_answer": "Yes, you need to write weekly logs every week during your internship until you complete the required hours for the credit. For undergraduate students, it's 150 hours for 4 credits of COMP690. For graduate students, the credit hour is roughly equal to 40 hours of internship work based on the number of credit hours you are enrolled in. After reaching the total hours required, it's recommended to continue with the weekly logs, but you don't need to submit logs for weeks you have not worked, like during a break.",
        "retrieval_context": ["Yes, you need to write weekly logs every week during your internship until you complete the required hours for the credit. For undergraduate students, it's 150 hours for 4 credits of COMP690. For graduate students, the credit hour is roughly equal to 40 hours of internship work based on the number of credit hours you are enrolled in. After reaching the total hours required, it's recommended to continue with the weekly logs, but you don't need to submit logs for weeks you have not worked, like during a break."]
    },
    {
        "question": "How many hours do I need to log?",
        "expected_answer": "You need to log 150 hours if you are taking COMP690. For graduate students, a credit hour is equal to 40 hours of internship work. For example, if you are enrolled in 3 credit hours of the Internship Experience class, then you must complete 120 hours of internship work.",
        "retrieval_context": ["For undergraduate students, you need to complete 150 hours for 4 credits of COMP690. For graduate students, a credit hour is roughly to 40 hours of internship work. For example, if you are enrolled in 3 credit hours of the Internship Experience class, then you must complete 120 hours of internship work."]
    },
    {
        "question": "Can I start my internship position before the Internship Experience course starts?",
        "expected_answer": "Yes, it is permissible to start your internship position before the Internship Experience course starts. However, you can only count up to 20% of the total internship hours required towards the course if you complete the remaining hours during the same semester.",
        "retrieval_context": ["It is permissible to start your internship position before the Internship Experience course starts. However, you can only count up to 20% of the total internship hours required towards the course if you complete the remaining hours during the same semester."]
    },
    {
        "question": "I just got an internship offer but the semester has already started, what should I do?",
        "expected_answer": "If you receive an internship offer after the semester has already started, you should contact the faculty internship coordinator, Professor Karen Jin, and inform her of the situation. Depending on the timing of your offer, you may be allowed to late add into the internship course or arrange with the employer for a later start date.",
        "retrieval_context": ["If you receive an internship offer after the semester has already started, you should contact the faculty internship coordinator, Professor Karen Jin, and inform her of the situation. Depending on the timing of your offer, you may be allowed to late add into the internship course or arrange with the employer for a later start date."]
    },
    {
        "question": "How do I contact CaPS office.",
        "expected_answer": "CaPS office has a website is https://manchester.unh.edu/careers/career-professional-success. You can also reach them by email unhm.career@unh.edu and by phone (603) 641-4394",
        "retrieval_context": ["The website for the CaPS office is https://manchester.unh.edu/careers/career-professional-success. Phone: (603) 641-4394 Email: unhm.career@unh.edu"]
    },
    {
        "question": "How many hours I can count if I start my internship work before starting the course?",
        "expected_answer": "you can only count up to 20% of the total internship hours required towards the course if you complete the remaining hours during the same semester.",
        "retrieval_context": ["It is permissible to start your internship position before the Internship Experience course starts. However, you can only count up to 20% of the total internship hours required towards the course if you complete the remaining hours during the same semester."]
    },
    {
        "question": "what is the oiss' website?",
        "expected_answer": "The website for the Office of International Students and Scholars (OISS) is https://www.unh.edu/global/international-students-scholars. You may also email them at oiss@unh.edu",
        "retrieval_context": ["The website for the Office of International Students and Scholars (OISS) is https://www.unh.edu/global/international-students-scholars Their email is oiss@unh.edu"]
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

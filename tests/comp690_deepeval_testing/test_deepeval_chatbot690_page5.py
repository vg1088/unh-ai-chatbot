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

# Define the test cases with expected answers for COMP690 Page 5
test_cases = [
    {"question": "What questions should be answered in the self-assessment section?", "expected_answer": "The self-assessment should address: What you learned, the relationship of the work to your major, the benefits to you, a comparison of theory (classroom learning) and practice (internship experience), how the project activities correlate with classroom knowledge, how the project will influence future career plans, a reflection on internship experiences (including skills needing development), and advice for a fellow student and/or faculty member."},
    {"question": "What comparison should be made in the self-assessment?", "expected_answer": "In the self-assessment for COMP690, you should compare your internship experiences with the learning objectives outlined in the syllabus. Assess how well you met these objectives, the skills you developed, and the challenges you faced during your internship. This reflection will help you evaluate your growth and the overall value of the experience."},
    {"question": "How should the project's influence on future career plans be discussed?", "expected_answer": "Discuss how the project activities and experiences will influence your future career plans."},
    {"question": "What reflections on the internship experience should be included?", "expected_answer": "In COMP690, your reflections on the internship experience should include thoughts on what you learned, how the experience contributed to your professional development, and any challenges you faced.."},
    {"question": "How long should the self-assessment section be (minimum)?", "expected_answer": "In the COMP690 syllabus, the self-assessment section should be a minimum of 2 pages long."},
    {"question": "What should be included in the conclusion section?", "expected_answer": "In the conclusion section of your COMP690 internship report, you should summarize the key insights and experiences gained during your internship. This includes reflecting on how the internship contributed to your professional development, any challenges you faced and how you overcame them, and the overall impact of the experience on your career goals."},
    {"question": "How long should the conclusion section be?", "expected_answer": "The COMP690 syllabus doesn’t specify an exact length for the conclusion section. However, it’s generally good practice for a conclusion to effectively summarize your main points, reflect on your internship experience, and provide insights or lessons learned."},
    {"question": "What is the required formatting for the internship report (spacing)?", "expected_answer": "For COMP690, the internship report should be formatted with double spacing."},
    {"question": "What is the required page range for the entire report (excluding title page, figures and tables)?", "expected_answer": "Between 6-8 pages."},
    {"question": "What font size is required for the report?", "expected_answer": "The font size required for the report in COMP690 is 12-point."},
    {"question": "What is the breakdown of grading criteria for the final report?", "expected_answer": "60% content, 20% grammar and mechanics, 20% format."},
    {"question": "What is the consequence of failing to meet the page requirements?", "expected_answer": "Up to a 30% deduction from the total report grade."},
    {"question": "What is the minimum percentage needed on the final report to pass the course?", "expected_answer": " The minimum percentage needed on the final report to pass COMP690 is typically 70%. "},
    {"question": "Should bullet points be used in the final report?", "expected_answer": "No, use full sentences."},
    {"question": "How are tables and figures to be handled in the report?", "expected_answer": "All tables and figures must be captioned."},
    {"question": "Can I use a font other than Times New Roman?", "expected_answer": "No, Times New Roman is specified."},
    {"question": "What is the minimum page length for the conclusion section?", "expected_answer": "1 full page."},
    {"question": "What additional skills should be identified for career readiness in the self-assessment?", "expected_answer": " In the COMP690 syllabus, students are encouraged to focus on both technical and soft skills in their self-assessment for career readiness. You may consider identifying skills such as communication, teamwork, problem-solving, adaptability, time management, and any specific technical skills related to your field of study. It's also beneficial to reflect on skills you've gained during your internship experience."},
    {"question": "What advice should be given to a fellow student and/or faculty member?", "expected_answer": "Advice related to the student's internship experience and lessons learned."},
    {"question": "What should be included in a reflection on the internship experience?", "expected_answer": "In a reflection on your internship experience for COMP690, you should include the following elements such as Overview of the Internship, Learning Outcomes, Challenges Faced, Connections to Coursework, Future Application, Personal Growth."},
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

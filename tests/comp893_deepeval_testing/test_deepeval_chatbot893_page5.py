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

# Define the test cases with expected answers for Page 5
test_cases = [
    {"question": "What should be included in my self-assessment?", "expected_answer": "Your self-assessment should address: what you learned, the project's relevance to your major, the benefits you gained, a comparison of theory (classroom learning) and practice (internship experience), how project activities correlated with classroom knowledge, how the project will influence your future career, reflections on your internship experience (including skills needing development), and advice for a fellow student or faculty member."},
    {"question": "What comparison should I make in the self-assessment?", "expected_answer": "Compare theory (classroom learning) and practice (your internship experiences)."},
    {"question": "How should I discuss the project's influence on my future career?", "expected_answer": "Explain how the project and internship experiences will affect your future career goals and plans."},
    {"question": "What reflections should be included in the self-assessment?", "expected_answer": "Describe what you learned, how it will be applied to your career, what additional skills you need to develop, and any advice you would offer to others, Describe how the experiences and skills acquired during the internship will apply to your professional goals."},
    {"question": "What's the minimum length for the self-assessment section?", "expected_answer": "For COMP893, the self-assessment section of your report should be 1 full page long, not including spacing, figures, and tables. Make sure to reflect on what you learned from your internship experience in that section."},
    {"question": "What should the conclusion section include?", "expected_answer": "A summary of key conclusions from your project experience."},
    {"question": "How long should the conclusion section be?", "expected_answer": "the conclusion section should be 1 full page long. Make sure it's single-spaced and follows the overall formatting requirements."},
    {"question": "What's the required spacing for the report?", "expected_answer": "you should use single spacing throughout the document. The report must be 6-8 pages long, not including the title page, figures, and tables."},
    {"question": "How many pages should the report be (excluding title page, figures, tables)?", "expected_answer": "It should be between 6 to 8 pages long, excluding the title page, figures, and tables. Make sure to format it with single-spaced lines, use size 12 Times New Roman font, and include no additional white space between paragraphs and sections."},
    {"question": "What font size is required?", "expected_answer": "you must use size 12 in Times New Roman font."},
    {"question": "What are the grading criteria for the report?", "expected_answer": "60% Content, 20% Grammar and Mechanics, 20% Format."},
    {"question": "What happens if my report is too short?", "expected_answer": "If your report is too short, you might face a penalty, and it's best to communicate with your instructor about the situation. They may provide you guidance on how to address the issue before submission."},
    {"question": "What's the minimum grade needed on the final report to pass?", "expected_answer": "you need to earn a minimum of 75% on your final report."},
    {"question": "Can I use bullet points in my report?", "expected_answer": "In your report, you should use full sentences instead of bullet points."},
    {"question": "How should I present tables and figures?", "expected_answer": "All tables and figures must be captioned."},
    {"question": "Can I use a different font than Times New Roman?", "expected_answer": "The formatting requirement specifies that you must use size 12 in Times New Roman font. So unfortunately, you can't use a different font than Times New Roman."},
    {"question": "What's the minimum length for the conclusion?", "expected_answer": "For the conclusion in final report, it should be at least one full page. This section should summarize the key conclusions derived from your project experience."},
    {"question": "What additional skills should I mention for career readiness?", "expected_answer": "Mention skills like learning new technologies, networking, or additional coursework."},
    {"question": "What kind of advice should I give to others?", "expected_answer": "Advice based on your internship experience and what you learned."},
    {"question": "What should my reflection on the internship experience cover?", "expected_answer": "What you learned, how it applies to your career goals, additional skills needed for career readiness, and any advice for others."}
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

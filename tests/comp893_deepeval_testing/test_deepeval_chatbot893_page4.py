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

# Define the test cases with expected answers for Page 4
test_cases = [
    {"question": "What is the policy on late submissions?", "expected_answer": "the late submission policy is very strict and applies only in exceptional cases, such as student illness, accidents, or emergencies that are properly documented. If you need to request a late submission, you must email the instructor prior to the deadline and explain the circumstances, providing evidence for why you could not meet the requirement"},
    {"question": "When would a late submission be accepted?", "expected_answer": "Late submissions are only accepted in exceptional cases, such as student illness, accidents, or emergencies. To be considered for a late submission, you must email the instructor before the deadline, explaining your circumstances and providing proper documentation."},
    {"question": "What needs to be in a late submission request?", "expected_answer": "For a late submission request You need to Send an email prior to the deadline, and Clearly explain the reasons for your late submission."},
    {"question": "What's on the title page of the final report?", "expected_answer": "Student's full name, internship start and finish dates, and project title."},
    {"question": "What should the executive summary include?", "expected_answer": "A concise overview of the project, including objectives, duration, and key outcomes."},
    {"question": "What should the introduction section cover?", "expected_answer": " In the introduction section of your COMP893 internship report, you should cover the Introduction of the project and provide background information, Clearly state the problem or challenge that the project addresses."},
    {"question": "What needs to be in the Project Objectives section?", "expected_answer": "Clearly stated objectives, including deliverables and their impact."},
    {"question": "How should the use of the Scrum framework be explained?", "expected_answer": "Describe how Scrum was adopted and implemented; discuss roles, responsibilities, and any adaptations."},
    {"question": "What does the self-assessment section need to include?", "expected_answer": "Answers to questions about what was learned, the project's relation to major studies, benefits, comparison of theory and practice, correlation with classroom knowledge, future career plans, reflection on the internship, and advice for others."},
    {"question": "How long should the executive summary be?", "expected_answer": "The executive summary of the internship project for COMP893 should provide a concise overview, including the project's objectives, duration, and key outcomes. While the specific length for the executive summary isn't explicitly stated in the documents, it is typically advisable for it to be brief—usually around one page or less, summarizing the essential points clearly."},
    {"question": "What formatting is required for the final report?", "expected_answer": "The report should be **6-8 pages** long (excluding the title page, figures, and tables), Use Size 12 Times New Roman font, There should be **no additional white space** between paragraphs and sections, All pages must be numbered, Finally save the report in **PDF format** for submission "},
    {"question": "How many pages are needed for the self-assessment?", "expected_answer": " Need a minimum of 3 full pages (not including spacing, figures, and tables). Be sure to use full sentences and avoid bullet points in your writing."},
    {"question": "How long does the entire final report need to be?", "expected_answer": "the final report should be between 6-8 pages long, not including the title page, figures, and tables. Make sure it is single-spaced, in 12-point Times New Roman font, with no extra white space between paragraphs and sections."},
    {"question": "What goes in the conclusion section?", "expected_answer": "In the conclusion section of your COMP893 internship report, you should include a summary of the key conclusions derived from your project experience. This should cover the main insights and takeaways you gained during the internship."},
    {"question": "What font and size should be used?", "expected_answer": "For your report, you should use Size 12 in Times New Roman font. The report needs to be single-spaced with no additional white space between paragraphs and sections."},
    {"question": "How should tables and figures be handled?", "expected_answer": "All tables and figures must be captioned."},
    {"question": "What are the grading criteria for the final report?", "expected_answer": "60%  for content, 20% for grammar and mechanics, 20% for format."},
    {"question": "What happens if the page requirements aren't met?", "expected_answer": " If the page requirements for your internship report aren't met, there could be a significant impact on your grade. Specifically, failing to fulfill the page requirements may result in up to a 30% deduction of the total report grade."},
    {"question": "What's the minimum grade needed on the final report to pass?", "expected_answer": "you need to earn a minimum of 75% on your final report."},
    {"question": "In what format should the final report be submitted?", "expected_answer": "The final report for COMP893 should include 6-8 pages long (not including the title page, figures, and tables), Written in Size 12 Times New Roman font, and the report must be saved in PDF format for submission."}
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

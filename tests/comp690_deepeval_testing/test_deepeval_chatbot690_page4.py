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

# Define the test cases with expected answers for COMP690 Page 4
test_cases = [
    {"question": "What is the policy regarding late submissions?", "expected_answer": "The policy is very strict and only applies in exceptional cases (illness, accident, emergencies) with proper documentation."},
    {"question": "Under what circumstances are late submissions considered?", "expected_answer": "Late submissions for COMP690 are typically considered under extenuating circumstances, such as personal emergencies or unforeseen events."},
    {"question": "What must be included in a request for a late submission?", "expected_answer": "For a late submission request in COMP690, you should include like, A brief explanation of the reason for the delay, Any supporting documentation, if applicable."},
    {"question": "What are the requirements for the final report title page?", "expected_answer": "The student's full name, internship start and finish dates, and the project title."},
    {"question": "What should be included in the executive summary of the final report?", "expected_answer": " In the executive summary of your final report for COMP690, you should include a brief overview of the following elements such as Objectives, Internship Experience, Learning Outcomes,Conclusion."},
    {"question": "What should be included in the introduction section of the final report?", "expected_answer": "In the introduction section of the final report for COMP690, you should typically include the following elements such as Purpose of the Report, Internship Overview, Goals and Objectives."},
    {"question": "What should be described in the Project Objectives section?", "expected_answer": "In the Project Objectives section for COMP690, you should describe the specific goals and outcomes you aim to achieve during your internship. This can include the skills you plan to develop, the projects you will be involved in, and how these objectives align with your overall learning and career goals."},
    {"question": "How should the use of the Scrum framework be explained in the report?", "expected_answer": "Explain how the Scrum framework was adopted and implemented, including roles, responsibilities, and any adjustments made."},
    {"question": "What should be included in the self-assessment section?", "expected_answer": "Answers to the questions: What you learned, the relationship of the work to your major, the benefits to you, comparison of theory and practice, the project's influence on your future career, reflections on the internship experience, and advice for fellow students/faculty."},
    {"question": "How long should the executive summary be?", "expected_answer": "In the COMP690 course, the executive summary should be approximately 1-2 pages long."},
    {"question": "What formatting is required for the final report?", "expected_answer": "Single-spaced, 6-8 pages (excluding title page, figures, tables), 12-point Times New Roman font, no extra space between paragraphs, all tables/figures captioned, page numbers, and submitted as a PDF."},
    {"question": "What is the minimum number of pages required for the self-assessment section?", "expected_answer": "Minimum 3 full pages (excluding spacing, figures, and tables)."},
    {"question": "What is the minimum number of full pages required for the final report?", "expected_answer": "Minimum 2 full pages for the executive summary section. The total report should be between 6-8 pages (excluding title page, figures, and tables)."},
    {"question": "What should be included in the conclusion section of the final report?", "expected_answer": "In the conclusion section of the final report for COMP690, you should summarize the key findings of your internship experience, reflect on what you learned, and discuss how the experience has impacted your personal and professional development. Additionally, consider mentioning any challenges you faced and how you overcame them, along with any recommendations for future interns. This is also a great place to express gratitude to your mentors or colleagues who supported you throughout your internship."},
    {"question": "What is the required font size and style for the final report?", "expected_answer": "For the final report in COMP690, the required font size is 12-point, and the preferred font style is Times New Roman."},
    {"question": "How should tables and figures be presented in the report?", "expected_answer": "All tables and figures must be captioned."},
    {"question": "What are the grading criteria for the final report?", "expected_answer": "The grading criteria for the final report in COMP690 typically include the following aspects: clarity and organization of the report, depth of analysis and critical thinking, completeness of the content, adherence to formatting guidelines, and the overall quality of writing, including grammar and style."},
    {"question": "What is the penalty for not meeting the page requirements?", "expected_answer": "Up to a 30% deduction from the total report grade."},
    {"question": "What is the minimum passing grade for the final report?", "expected_answer": " In COMP690, the minimum passing grade for the final report is a C"},
    {"question": "What file format should the final report be submitted in?", "expected_answer": "The final report for COMP690 should be submitted in PDF format."}
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

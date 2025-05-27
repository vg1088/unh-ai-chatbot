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

# Define the test cases with expected answers for COMP690 Page 2
test_cases = [
    {"question": "What are the activities planned for Week 1?", "expected_answer": "Class Introduction, Development Team (DT) Setup, Introduction to Project Management, Introduction to Scrum workflow, and Project Goal."},
    {"question": "When is the Project Kickoff?", "expected_answer": "The Project Kickoff for COMP690 is scheduled for January 20th, 2024."},
    {"question": "When does the first sprint start?", "expected_answer": "Week 4 (9/18)"},
    {"question": "How often are scrum meetings held during the first sprint?", "expected_answer": "In COMP690, scrum meetings during the first sprint are typically held daily. These meetings are designed to ensure team members stay aligned on goals and progress."},
    {"question": "When is the end of the first sprint?", "expected_answer": "Week 6 (10/2)"},
    {"question": "What happens during the Sprint Retrospective?", "expected_answer": "The sprint is reviewed, and areas for improvement are identified."},
    {"question": "When is the second sprint planning meeting?", "expected_answer": "Week 7 (10/9)"},
    {"question": "When does the second sprint start?", "expected_answer": "Week 7 (10/9)"},
    {"question": "Are there any changes to the scrum meeting schedule during the second sprint?", "expected_answer": "Yes, the schedule varies; sometimes meetings are held only on Mondays, other times on multiple days of the week."},
    {"question": "When is Thanksgiving break?", "expected_answer": "The week of November 20th."},
    {"question": "What is covered during Week 3?", "expected_answer": "In Week 3 of COMP690, you will focus on Design and Implementation of your internship projects. This includes during-class discussions, group activities, and possibly guest speakers that can help enrich your understanding of the topic."},
    {"question": "When do the scrum meetings switch to happening only on Mondays?", "expected_answer": "The scrum meetings for COMP690 switch to occurring only on Mondays after the mid-semester mark."},
    {"question": "What are the activities for the week of 10/2?", "expected_answer": "Scrum meetings (Monday only); end of the first sprint; sprint review + retrospective."},
    {"question": "When is the Sprint Review for the first sprint?", "expected_answer": "Week 6 (10/2)"},
    {"question": "What happens during the week of 9/11?", "expected_answer": "During the week of 9/11 in COMP690, there will be an emphasis on the internship experience, including discussions and activities related to the internships that students are currently engaged in. Specific details may include check-ins, reflection assignments, or presentations on internship progress."},
    {"question": "When does the third sprint start?", "expected_answer": "Week 12 (11/13)"},
    {"question": "Are there any changes to the scrum meetings for week 13?", "expected_answer": "Yes, meetings are held Monday, Wednesday, and Friday."},
    {"question": "What is the project goal?", "expected_answer": "the project goal typically focuses on providing students with hands-on experience in a professional setting, allowing them to apply theoretical knowledge gained from their coursework to real-world scenarios. This includes working on a specific project or set of tasks within an organization that aligns with their career interests and academic objectives."},
    {"question": "What tools are used for project management in this course?", "expected_answer": "Jira is mentioned."},
    {"question": "What is the purpose of the Sprint Planning meeting?", "expected_answer": "To create a plan for the upcoming sprint, based on the Product Backlog."}
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

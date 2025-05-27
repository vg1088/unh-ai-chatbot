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

# Define the test cases with expected answers
test_cases = [
    {"question": "What happens in week 1?", "expected_answer": "In week 1 of COMP893, the course likely focuses on an introduction to the internship experience and course expectations, so if you're looking for precise activities like Class introduction, team setup, project management intro, scrum workflow intro, and project goal it may be best to check the syllabus directly."},
    {"question": "When is the project kickoff?", "expected_answer": "The project kickoff for COMP893 is scheduled for Week 2, on September 4th"},
    {"question": "When does the first sprint begin?", "expected_answer": "The first sprint for COMP893 begins on September 18th, when the development starts along with Scrum meetings"},
    {"question": "How often are scrum meetings during the first sprint?", "expected_answer": "During the first sprint of COMP893, scrum meetings are held three times a week: on Monday, Wednesday, and Friday ."},
    {"question": "When does the first sprint end?", "expected_answer": "The first sprint for COMP893 ends on October 2nd, which is also when the Sprint Review and Sprint Retrospective will take place."},
    {"question": "What's involved in a sprint retrospective?", "expected_answer": "Reviewing the sprint and identifying areas for improvement."},
    {"question": "When is the second sprint planning meeting?", "expected_answer": "The second sprint planning meeting for COMP893 is scheduled for October 9th during Week 7."},
    {"question": "When does the second sprint start?", "expected_answer": "The second sprint for COMP893 starts on October 9th, during Week 7"},
    {"question": "Does the scrum meeting schedule change during the second sprint?", "expected_answer": " Yes, the scrum meeting schedule does change during the second sprint. In Week 7, after the 2nd Sprint Planning meeting on 10/9, scrum meetings are held on Fridays. In Week 8, the schedule changes to meetings on Wednesday and Friday. Then in Week 9, meetings occur on Monday, Wednesday, and Friday. So, the schedule evolves throughout the sprints."},
    {"question": "When is Thanksgiving Break?", "expected_answer": "Thanksgiving Break for COMP893 is from November 20th to November 27th"},
    {"question": "What's the focus of week 3?", "expected_answer": "Environment setup (Jira), creating the project backlog, user stories, tasks, and bugs; integration with source control, team communication, and a sprint planning meeting."},
    {"question": "When are scrum meetings held only on Mondays?", "expected_answer": "scrum meetings are held on Mondays only during Week 14 (starting on November 27) and Week 15 (starting on December 4)"},
    {"question": "What happens during week 10?", "expected_answer": "Scrum meetings (Monday, Wednesday, Friday)."},
    {"question": "When is the sprint review for the first sprint?", "expected_answer": "The Sprint Review for the first sprint in COMP893 is scheduled for the end of Week 6, specifically on October 2nd."},
    {"question": "What are the activities for Week 9?", "expected_answer": "Scrum meetings (Monday, Wednesday, Friday)."},
    {"question": "When does the third sprint start?", "expected_answer": "The third sprint for COMP893 starts on November 13, during Week 12"},
    {"question": "How often are scrum meetings in week 13?", "expected_answer": "scrum meetings are held on Monday, Wednesday, and Friday"},
    {"question": "What is the project goal?", "expected_answer": "The project goal for COMP893 involves working collaboratively on a real-world internship project that allows students to apply the Scrum framework. The focus is to introduce a project by discussing its background, objectives, and the significance of using Scrum. Additionally, students will set up the project environment, create backlogs, and plan sprints, with an end goal of delivering a final report that encapsulates their experiences and outcomes."},
    {"question": "What tools are used for project management?", "expected_answer": "Jira is used for project management."},
    {"question": "What is the purpose of sprint planning?", "expected_answer": "To create a plan for the upcoming sprint based on the Product Backlog."}
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

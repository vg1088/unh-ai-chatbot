import pytest
import requests
import re
from sentence_transformers import SentenceTransformer, util

# Load the model for embedding
model = SentenceTransformer('all-MiniLM-L6-v2')

# Define the test cases with expected answers
test_cases_page2 = [
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

# Send question to chatbot and get response
def get_chatbot_response(session, question):
    try:
        response = session.post("http://localhost:1896/llm_response", data={"message": question})
        response.raise_for_status()
        return response.text.strip()
    except requests.exceptions.RequestException as e:
        print(f"Error connecting to chatbot API: {e}")
        return None

# Preprocess the actual response to focus on the main content
def preprocess_response(response):
    # Remove polite phrases and focus on the core answer
    response = re.sub(r"(?i)(if you have any more questions.*|feel free to ask.*|let me know if.*)", "", response)
    response = response.strip()
    return response

# Semantic similarity check
def check_response_semantically(expected, actual, threshold=0.5):
    if not actual:
        return False, 0  # Return failed check if no response
    
    # Embed both expected and actual responses
    expected_embedding = model.encode(expected, convert_to_tensor=True)
    actual_embedding = model.encode(actual, convert_to_tensor=True)
    
    # Calculate cosine similarity between embeddings
    similarity = util.cos_sim(expected_embedding, actual_embedding).item()
    
    # Return whether similarity meets threshold
    return similarity >= threshold, similarity

# Explicitly set the context to COMP893 for each test case
def set_course_context(session):
    context_question = "I am asking about COMP893"
    get_chatbot_response(session, context_question)

@pytest.mark.parametrize("test_case", test_cases_page2)
def test_chatbot_responses_page2(test_case):
    # Create a new session per test
    session = requests.Session()

    # Set context explicitly for each test
    set_course_context(session)

    question = test_case["question"]
    expected_answer = test_case["expected_answer"]

    # Get chatbot response for the actual question
    actual_response = get_chatbot_response(session, question)
    print(f"Actual response for '{question}': {actual_response}")  # Print the actual response

    # Preprocess actual response
    actual_response_processed = preprocess_response(actual_response)

    # Check response semantically
    passed, similarity_score = check_response_semantically(expected_answer, actual_response_processed)
    result = "pass" if passed else "fail"

    # Log similarity for debugging
    print(f"Similarity score for '{question}': {similarity_score}")

    # Assert that the result passed
    assert passed, f"Failed for question: {question}, similarity: {similarity_score}"

import pytest
import requests
import re
from sentence_transformers import SentenceTransformer, util

# Load the model for embedding
model = SentenceTransformer('all-MiniLM-L6-v2')

# Define the test cases with expected answers for COMP690 Page 1
test_cases_comp690_page1 = [
    {"question": "What time the class schedule?", "expected_answer": "The M2 section is Wednesday from 9:10 AM to 12:00 PM, and the M3 section is Wednesday from 1:10 PM to 4:00 PM."},
    {"question": "Office hours?", "expected_answer": "Professor Karen Jin's office hours are Monday from 1:00 PM to 4:00 PM and Friday from 9:00 AM to 12:00 PM."},
    {"question": "Where’s the class?", "expected_answer": "Room P142."},
    {"question": "How many credits?", "expected_answer": "4 credits."},
    {"question": "Is there a Zoom link?", "expected_answer": "Yes, the Zoom link for Professor Karen Jin’s office hours is Join our Cloud HD Video Meeting."},
    {"question": "Who's the instructor?", "expected_answer": "Professor Karen Jin, Associate Professor in the Department of Applied Engineering and Sciences."},
    {"question": "When is Karen Jin's office hours?", "expected_answer": "Monday 1:00 PM to 4:00 PM and Friday 9:00 AM to 12:00 PM."},
    {"question": "What time does the M2 section start on Wednesday?", "expected_answer": "9:10 AM."},
    {"question": "How can I schedule an appointment with Professor Jin?", "expected_answer": "Email Professor Jin at karen.jin@unh.edu."},
    {"question": "Where is Professor Jin's office located?", "expected_answer": "Room 139, Pandora Mill building."},
    {"question": "What is the professor's email address?", "expected_answer": "karen.jin@unh.edu"},
    {"question": "What room is the class in?", "expected_answer": "Room P142."},
    {"question": "Can I meet the professor on Monday afternoon?", "expected_answer": "Yes, during office hours from 1:00 PM to 4:00 PM."},
    {"question": "Will Professor Jin be available on Fridays?", "expected_answer": "Yes, from 9:00 AM to 12:00 PM."},
    {"question": "How do I join the professor's Zoom for office hours?", "expected_answer": "Use this link: Join our Cloud HD Video Meeting."},
    {"question": "Can I make an appointment outside of office hours?", "expected_answer": "Yes, email Professor Jin at karen.jin@unh.edu."},
    {"question": "Can I email the professor to set up a meeting?", "expected_answer": "Yes, email her at karen.jin@unh.edu."},
    {"question": "What kind of projects will we do in this internship?", "expected_answer": "Team-based projects involving real-world IT products, processes, or services with external stakeholders."},
    {"question": "Wht is prof jin's email?", "expected_answer": "karen.jin@unh.edu"},
    {"question": "When are ofice hurs for prof Jin?", "expected_answer": "Monday 1:00 PM to 4:00 PM and Friday 9:00 AM to 12:00 PM."}
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

# Explicitly set the context to COMP690 for each test case
def set_course_context(session):
    context_question = "I am asking about COMP690"
    get_chatbot_response(session, context_question)

@pytest.mark.parametrize("test_case", test_cases_comp690_page1)
def test_chatbot_responses_comp690_page1(test_case):
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

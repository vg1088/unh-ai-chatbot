import pytest
import requests
import re
from sentence_transformers import SentenceTransformer, util

# Load the model for embedding
model = SentenceTransformer('all-MiniLM-L6-v2')

# Define the test cases with expected answers
test_cases = [
    {"question": "What's the course name?", "expected_answer": "COMP 893 Team Project Internship"},
    {"question": "How many credits is this course?", "expected_answer": "COMP893 offers 1 to 3 credits"},
    {"question": "When is this course offered?", "expected_answer": "COMP893 is offered in the Fall 2024 term. The class location is in Room P142, with two sections on Wednesdays: the first from 9:10 AM to 12 PM and the second from 1:10 PM to 4 PM."},
    {"question": "Where are classes held?", "expected_answer": "Classes for COMP893 are held in Room P142."},
    {"question": "What time is the M1 section on Wednesdays?", "expected_answer": "The M1 section of COMP893 on Wednesdays meets from 9:10 AM to 12:00 PM."},
    {"question": "What time is the M2 section on Wednesdays?", "expected_answer": "The M2 section of COMP893 on Wednesdays meets from 1:10 PM to 4:00 PM"},
    {"question": "Who is the instructor?", "expected_answer": "The instructor for COMP893 is Karen Jin, who is an Associate Professor in the Department of Applied Engineering and Sciences."},
    {"question": "What's the instructor's title?", "expected_answer": "Associate Professor"},
    {"question": "What department is the professor in?", "expected_answer": "The professor for COMP893 is Karen Jin, who is an Associate Professor in the Department of Applied Engineering and Sciences."},
    {"question": "Where is the professor's office?", "expected_answer": "The professor's office is located in Room 139 of the Pandora Mill building"},
    {"question": "What is the professor's Zoom link?", "expected_answer": "https://unh.zoom.us/j/4858446046"},
    {"question": "What is the professor's email?", "expected_answer": "karen.jin@unh.edu"},
    {"question": "What are the professor's office hours?", "expected_answer": "The professor's office hours for COMP893 are on Monday from 1-4 PM and Friday from 9 AM to noon. You can meet with them in person at Room 139 in the Pandora Mill building"},
    {"question": "Are office hours available online?", "expected_answer": " Yes, office hours for COMP893 are available both in person and over Zoom. They are held on Monday from 1-4 PM and Friday from 9 AM to noon. You can use this Zoom link to join: [Zoom Link](https://unh.zoom.us/j/4858446046)."},
    {"question": "How do I schedule an appointment with the professor?", "expected_answer": "You can schedule an appointment with Professor Karen Jin by emailing her at karen.jin@unh.edu. She holds office hours in person on Monday from 1-4 PM and Friday from 9 AM-noon"},
    {"question": "What building is the class in?", "expected_answer": "The COMP893 class is held in Room P142."},
    {"question": "What's the course description?", "expected_answer": "The internship course provides experiential learning experience through placement in team projects. This hands-on experience allows students to gain practical skills and insights into the field of computing. By working on a collaborative project with external stakeholders, they will contribute to the development of real-world information technology products, processes, or services, and understand the challenges involved in implementing technology solutions in a professional setting."},
    {"question": "Is there a phone number for the professor?", "expected_answer": "I don't have a specific phone number for the professor, but you can reach out to Karen Jin via email at karen.jin@unh.edu. You can also visit her during office hours in Room 139, Pandora Mill building, or connect with her over Zoom."},
    {"question": "What is the professor's full title?", "expected_answer": "Associate Professor, Department of Applied Engineering and Science"}
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

@pytest.mark.parametrize("test_case", test_cases)
def test_chatbot_responses(test_case):
    # Create a new session per test
    session = requests.Session()
    
    # Set context explicitly for each test
    set_course_context(session)
    
    question = test_case["question"]
    expected_answer = test_case["expected_answer"]
    
    # Get chatbot response for the actual question
    actual_response = get_chatbot_response(session, question)
    print(f"Actual response for '{question}': {actual_response}")  # Print the actual response

    # Check response semantically
    passed, similarity_score = check_response_semantically(expected_answer, actual_response)
    result = "pass" if passed else "fail"

    # Log similarity for debugging
    print(f"Similarity score for '{question}': {similarity_score}")

    # Assert that the result passed
    assert passed, f"Failed for question: {question}, similarity: {similarity_score}"
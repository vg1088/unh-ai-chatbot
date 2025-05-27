import pytest
import requests
import re
from sentence_transformers import SentenceTransformer, util

# Load the model for embedding
model = SentenceTransformer('all-MiniLM-L6-v2')

# Define the test cases with expected answers for chatbox_doc Page 3
test_cases_page3 = [
    {"question": "Can I use a previous internship to fulfill the course requirements?", "expected_answer": "No, you must have an internship while enrolled in the course."},
    {"question": "I did an internship last summer; can I use that?", "expected_answer": "No, you must have an active internship during the course."},
    {"question": "How do I register my internship experience on Handshake?", "expected_answer": "Log in, go to 'Experiences,' click 'Request an Experience,' complete the form, get approval from your supervisor and instructor."},
    {"question": "How do I add my internship to Handshake?", "expected_answer": "Follow the steps to 'Request an Experience' on Handshake, ensuring supervisor and instructor approval."},
    {"question": "What information do I need to put in Handshake?", "expected_answer": " In Handshake, you'll need to provide details such as your resume, cover letter, internship preferences, and any relevant work experience. It's also helpful to include your skills, education, and a professional profile picture. This will make it easier for employers to find you and for you to apply to internship opportunities."},
    {"question": "Where do I find the 'Request an Experience' form on Handshake?", "expected_answer": "In the 'Experiences' section of your Handshake account."},
    {"question": "Who needs to approve my internship on Handshake?", "expected_answer": "Your internship on Handshake needs to be approved by the faculty internship coordinator."},
    {"question": "How many learning objectives should I include?", "expected_answer": "At least three well-developed learning objectives."},
    {"question": "Where can I get help with Handshake?", "expected_answer": "You can get help with Handshake by visiting the Career Services office at your institution, or you can check the support section on the Handshake website. They often have resources like guides and FAQs to assist you."},
    {"question": "Do I have to do anything else besides the internship itself?", "expected_answer": "Yes, you must complete class requirements (attend meetings, submit logs, final report, presentations)."},
    {"question": "What are the requirements for this internship class?", "expected_answer": "Attend all class meetings, submit weekly logs, complete a final report, and give progress presentations."},
    {"question": "How many class meetings do I need to attend?", "expected_answer": "All scheduled meetings."},
    {"question": "How often are weekly logs due?", "expected_answer": "Every week you work at your internship."},
    {"question": "Are there any presentations required?", "expected_answer": "Yes, progress presentations during the class."},
    {"question": "What's involved in the final report?", "expected_answer": "The details are in the specific course syllabus."},
    {"question": "What are all the requirements for credit in this course?", "expected_answer": "Attending all class meetings, submitting weekly logs, completing a final internship report, and giving progress presentations."},
    {"question": "Is attending every class meeting mandatory?", "expected_answer": "Attending every class meeting is typically required for the internship courses."},
    {"question": "What is the format for weekly logs?", "expected_answer": "The specific format is in the course syllabus."},
    {"question": "What kind of internship report is required?", "expected_answer": "For the internship report, you typically need to provide a final document that summarizes your internship experience, including the tasks you undertook, skills developed, and lessons learned during your time at the organization."},
    {"question": "How many presentations should I expect to give?", "expected_answer": "The syllabus will have the details; the phrasing implies more than one."}
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

# Explicitly set the context to general internship questions for each test case
def set_course_context(session):
    context_question = "I am asking about general internship questions"
    get_chatbot_response(session, context_question)

@pytest.mark.parametrize("test_case", test_cases_page3)
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

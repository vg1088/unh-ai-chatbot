import pytest
import requests
import re
from sentence_transformers import SentenceTransformer, util

# Load the model for embedding
model = SentenceTransformer('all-MiniLM-L6-v2')

# Define the test cases with expected answers for COMP690 Page 6
test_cases_comp690_page6 = [
    {"question": "What is UNH's policy on academic integrity?", "expected_answer": "The UNH Academic Integrity Policy is available online; the syllabus provides a link to this policy."},
    {"question": "What is the policy regarding mandatory reporting of sexual violence or harassment?", "expected_answer": "Faculty members are required to report incidents of sexual violence or harassment to the university's Title IX Coordinator."},
    {"question": "How can I report sexual violence or harassment confidentially?", "expected_answer": "Contact the SHARPP Center for Interpersonal Violence Awareness, Prevention, and Advocacy."},
    {"question": "What resources are available for confidential support at UNH Manchester?", "expected_answer": "SHARPP Extended Services Coordinator (in person and via Zoom), YWCA NH, and the Mental Health Center of Greater Manchester."},
    {"question": "What are the contact details for the UNH Manchester Title IX Deputy Intake Coordinator?", "expected_answer": "The syllabus lists Lisa Enright's email address (lisa.enright@unh.edu) and phone number."},
    {"question": "What resources are available at the UNH Manchester library?", "expected_answer": "Assistance with research and access to online library resources."},
    {"question": "How can I contact the UNH Manchester library?", "expected_answer": "You can contact the library at 603-641-4173 or unhm.library@unh.edu."},
    {"question": "How can I make a research appointment with a librarian?", "expected_answer": "The syllabus provides a link to instructions on making a research appointment."},
    {"question": "How can I use the library search box?", "expected_answer": "The syllabus provides a link to instructions on using the library search box."},
    {"question": "How can I reserve a study room?", "expected_answer": "The syllabus provides a link to instructions on reserving a study room."},
    {"question": "Where can I find resources for citing sources?", "expected_answer": "The syllabus provides a link to resources for citing sources."},
    {"question": "What is the website for the UNH Manchester Library?", "expected_answer": "The syllabus provides the website for the UNH Manchester Library."},
    {"question": "What resources are available for evaluating sources?", "expected_answer": "The syllabus provides a link to resources for evaluating sources."},
    {"question": "What is the phone number of the SHARPP Center?", "expected_answer": "(603) 862-7233/TTY (800) 735-2964"},
    {"question": "What resources are available to report bias, discrimination, or harassment?", "expected_answer": "Contact the Civil Rights & Equity Office at UNH."},
    {"question": "What are the contact details for the UNH Title IX Coordinator?", "expected_answer": "The syllabus lists Bo Zaryckyj's email address (Bo.Zaryckyj@unh.edu) and phone number."},
    {"question": "What is the name of the app that provides access to reporting options and resources?", "expected_answer": "uSafeUS"},
    {"question": "Where can I find the UNH Academic Integrity Policy?", "expected_answer": "The syllabus includes a link to the UNH Academic Integrity Policy."},
    {"question": "Where can I find more information about what happens when I report an incident?", "expected_answer": "The syllabus mentions a page with information on student reporting options."},
    {"question": "What is the phone number for the 24-hour NH Domestic Violence Hotline?", "expected_answer": "1-866-644-3574"}
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

@pytest.mark.parametrize("test_case", test_cases_comp690_page6)
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

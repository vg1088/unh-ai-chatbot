import pytest
import requests
import re
from sentence_transformers import SentenceTransformer, util

# Load the model for embedding
model = SentenceTransformer('all-MiniLM-L6-v2')

# Define the test cases with expected outcomes for chatbox_doc Page 6
test_cases_page6 = [
    {"question": "What is Note 1 referring to?", "expected_answer": "The note explains that for COMP892 (applied research), the student needs a total of 3 credits which may be taken over multiple semesters and requires a tech job (part-time or full-time)."},
    {"question": "What does Note 2 mean?", "expected_answer": "If COMP890 has already been taken, then only 2 credits total are needed for COMP891 or COMP892; they can be taken over multiple semesters."},
    {"question": "What does Note 3 mean?", "expected_answer": "COMP893 must be finished in one semester and requires 2 credits if COMP890 has already been taken."},
    {"question": "Can a graduate student take COMP890 before completing 9 credits?", "expected_answer": "No, the flowchart shows it requires 9 credits completed first."},
    {"question": "Can an undergraduate student take the internship course before finding an internship?", "expected_answer": "Yes, but only by using the applied research option (COMP690) if they have a tech job."},
    {"question": "Can a graduate student take the internship course before completing the required credits?", "expected_answer": "No, according to the flowchart."},
    {"question": "Are there credit requirements for each course?", "expected_answer": "Yes, the notes on the chart detail credit requirements for each course (COMP890, COMP891, COMP892, COMP893). The specific number of credits depends on prerequisites and whether it is a thesis or project-based program."},
    {"question": "Can any of the courses be taken over multiple semesters?", "expected_answer": "Yes, Notes 1 and 2 on the chart specify that some courses can span multiple semesters."}
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

@pytest.mark.parametrize("test_case", test_cases_page6)
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

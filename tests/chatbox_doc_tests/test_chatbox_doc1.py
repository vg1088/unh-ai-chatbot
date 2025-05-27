import pytest
import requests
import re
from sentence_transformers import SentenceTransformer, util

# Load the model for embedding
model = SentenceTransformer('all-MiniLM-L6-v2')

# Define the test cases with expected answers for chatbox_doc Page 1
test_cases_page1 = [
    {"question": "What computing majors require internships?", "expected_answer": "Generally, majors like Computer Science, Computer Engineering, and Information Technology often require internships to help students gain practical experience. You may also want to refer to your program's syllabus or official guidelines for specific information about internship requirements."},
    {"question": "Is there an internship option for Cybersecurity Engineering students?", "expected_answer": "Yes, Cybersecurity Engineering students can pursue internships as part of their academic requirements."},
    {"question": "Who is the internship coordinator?", "expected_answer": "The internship coordinator is Karen Jin. You can reach her at karen.jin@unh.edu for any internship-related questions or concerns."},
    {"question": "Where is Karen Jin's office located?", "expected_answer": "Room 139, Pandora Mill building."},
    {"question": "What is Karen Jin's email address?", "expected_answer": "Karen.Jin@unh.edu"},
    {"question": "How can I contact the CaPS office?", "expected_answer": "You can contact them via phone at (603) 641-4394 or visit their website: Career and Professional Success"},
    {"question": "What is the CaPS website address?", "expected_answer": "Career and Professional Success"},
    {"question": "What is the phone number for the CaPS office?", "expected_answer": "(603) 641-4394"},
    {"question": "What is the OISS website?", "expected_answer": "International Students & Scholars"},
    {"question": "What is the OISS email address?", "expected_answer": "oiss@unh.edu"},
    {"question": "What undergraduate internship courses are available?", "expected_answer": "COMP690"},
    {"question": "What is COMP690 about?", "expected_answer": "It's an internship experience course with applied research and team project options."},
    {"question": "Does COMP690 offer a team project option?", "expected_answer": "Yes."},
    {"question": "What is COMP890?", "expected_answer": "A graduate-level internship course: Internship and Career Planning."},
    {"question": "When is COMP890 offered?", "expected_answer": "Fall and spring semesters."},
    {"question": "How many credits is COMP890?", "expected_answer": "1 credit."},
    {"question": "What is COMP891?", "expected_answer": "A graduate-level internship course: Internship Practice."},
    {"question": "What are the graduate-level internship courses?", "expected_answer": "The graduate-level internship courses are COMP690 and COMP893."},
    {"question": "What internship classes are there?", "expected_answer": "For undergraduates: COMP690. For graduates: COMP890 and COMP891."},
    {"question": "Tell me about the graduate internship program.", "expected_answer": "Graduate students can choose between COMP890 (Internship and Career Planning) and COMP891 (Internship Practice)."}
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

@pytest.mark.parametrize("test_case", test_cases_page1)
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

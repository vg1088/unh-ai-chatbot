import pytest
import requests
import re
from sentence_transformers import SentenceTransformer, util

# Load the model for embedding
model = SentenceTransformer('all-MiniLM-L6-v2')

# Define the test cases with expected answers for chatbox_doc Page 5
test_cases_page5 = [
    {"question": "What must be included in my internship offer letter?", "expected_answer": "Your internship offer letter should typically include the following key details like Position Title,Start and End Dates, Compensation, Work Schedule, Location, Responsibilities, Conditions of Employment."},
    {"question": "What dates should be on my internship offer letter?", "expected_answer": "The precise start and end dates of your employment during the semester."},
    {"question": "When can my internship employment begin?", "expected_answer": "Your internship employment can typically begin as soon as you have received approval from your course instructor and have completed any necessary prerequisites. However, specific starting dates can vary depending on the program and the timeline you set with your employer."},
    {"question": "Can I work between semesters?", "expected_answer": "No, there's approximately a four-week period between semesters where you cannot work."},
    {"question": "How do I request CPT authorization?", "expected_answer": "Meet with Professor Karen Jin, log in to eOISS, complete the CPT form, upload required documents, and wait for approval."},
    {"question": "Who should I contact about CPT authorization?", "expected_answer": " For CPT (Curricular Practical Training) authorization, you should contact your designated school official (DSO) or the international student office at your institution. They can provide you with the necessary information and steps to obtain CPT authorization."},
    {"question": "Where do I find the CPT authorization form?", "expected_answer": " You can find the CPT (Curricular Practical Training) authorization form by checking with your designated school official (DSO) or the international student office at your institution. It's typically available on their website or through their office."},
    {"question": "What information goes on the CPT form?", "expected_answer": "The CPT (Curricular Practical Training) form typically requires the following information: Personal Information, Program Details, Internship Information, Dates of Employment, Signatures."},
    {"question": "What documents need to be uploaded with the CPT form?", "expected_answer": "A copy of your job offer letter, Your current I-20 form, A completed CPT application form from your school, Any other documents specified by your academic department."},
    {"question": "What should the employer letter include?", "expected_answer": "The employer letter should include the following information: Employer's Information, Internship Details, Supervision and Mentorship, Working Hours, Offer of Internship."},
    {"question": "What is acceptable proof of course registration for CPT?", "expected_answer": "Acceptable proof of course registration for CPT (Curricular Practical Training) typically includes documents such as your course registration confirmation or a copy of your official class schedule."},
    {"question": "What happens after I submit the CPT form?", "expected_answer": "Your university's International Student Office will review your CPT application to ensure it meets all requirements, You’ll receive notification regarding the approval or denial of your CPT request,If approved, you may need to complete additional steps, such as updating your I-20."},
    {"question": "How long does the CPT approval process take?", "expected_answer": "The document doesn't give an exact time, but suggests submitting 7-10 days in advance."},
    {"question": "When should I submit my CPT application?", "expected_answer": "7-10 days before your internship start date."},
    {"question": "When can I start working after CPT approval?", "expected_answer": "After your CPT (Curricular Practical Training) approval, you can begin working on the start date listed on your CPT authorization. Make sure to adhere to the dates specified, and you cannot start any work before that date."},
    {"question": "What if I don't get my I-20 in time?", "expected_answer": "If you don't receive your I-20 in time, I recommend that you reach out to your school’s international student office or the designated school official (DSO) for guidance. They can provide specific advice on what steps to take and how it may affect your ability to start your internship."},
    {"question": "What is the timeline for taking the internship course as an undergrad?", "expected_answer": "You can take it anytime you have an internship; if you can't find one, the group project option in COMP690 is available."},
    {"question": "Can I take the internship course before I have an internship?", "expected_answer": "Yes, but only if you use the applied research option with a part-time or full-time tech job (COMP690)."},
    {"question": "What if I can't find an internship by my last semester?", "expected_answer": "You can complete the group project option in COMP690."},
    {"question": "Where can I find the timeline for graduate students?", "expected_answer": "The document states that it's in a chart below the paragraph; it should be on the next page."}
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

@pytest.mark.parametrize("test_case", test_cases_page5)
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

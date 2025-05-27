import pytest
import requests
import re
from sentence_transformers import SentenceTransformer, util

# Load the model for embedding
model = SentenceTransformer('all-MiniLM-L6-v2')


# Define the test cases with expected answers for Page 3
test_cases_page3 = [
    {"question": "What are the components of the final grade?", "expected_answer": "Class Attendance (10%), Sprint Grade (60%), Homework (10%), and Final Project Report (20%)."},
    {"question": "How much is class attendance worth?", "expected_answer": "10% of the final grade."},
    {"question": "How is the Sprint Grade calculated?", "expected_answer": "Teamwork Grade multiplied by Sprint Grade."},
    {"question": "What determines the Teamwork Grade?", "expected_answer": "Peer evaluation for each of the three sprints; detailed rubrics are to be announced (TBA)."},
    {"question": "What determines the Sprint Grade?", "expected_answer": "The overall Sprint Grade is calculated using the formula: **Sprint Grade = Teamwork Grade * Sprint Grade**. Additionally, class attendance (10%) and homework (30%) also contribute to your final grade."},
    {"question": "What percentage is the final project report worth?", "expected_answer": "The final project report in COMP893 is worth 20% of your overall grade."},
    {"question": "Where can I find the final project report format?", "expected_answer": "You can find the final project report format in Appendix A of the COMP893 syllabus. The report should include several sections, such as a title page, conclusions, and a self-assessment of the project experience."},
    {"question": "What's the policy on late submissions?", "expected_answer": "the policy on late submissions is quite strict. Late submissions may only be granted in exceptional cases such as illness, accidents, or emergencies, provided that these circumstances are properly documented. The student must email the instructor prior to the deadline and explain the situation, along with providing evidence to support their request."},
    {"question": "When might a late submission be considered?", "expected_answer": "Only if the student emails before the deadline, explains the circumstances, and provides evidence."},
    {"question": "What is UNH's attendance policy?", "expected_answer": "Students are responsible for attending scheduled meetings and are expected to abide by the University Policy on Attendance."},
    {"question": "What if I can't make a scheduled meeting?", "expected_answer": "Email the instructor beforehand, explain the situation, and request to be excused. Schedule a meeting to update the instructor."},
    {"question": "How many hours per week should I expect to spend on this course?", "expected_answer": "A minimum of 45 hours per credit per term."},
    {"question": "What should I do if I'm sick and can't come to class?", "expected_answer": "If you're sick and can't make it to class, it's important to email your instructor before the class meeting to explain your circumstances and request to be excused."},
    {"question": "What happens if I don't follow the late submission rules?", "expected_answer": "You may receive no credit for the assignment."},
    {"question": "Are there specific rubrics for teamwork evaluation?", "expected_answer": "teamwork evaluation is based on peer evaluations for each of the three sprints. The detailed rubrics for evaluating teamwork will be announced later. Your overall Sprint Grade will be calculated using the Teamwork Grade along with the Sprint Grade related to the technical aspect of the product and team project management."},
    {"question": "Are there specific rubrics for sprint evaluation?", "expected_answer": "Yes, there are specific rubrics for sprint evaluation, This is based on peer evaluation for each of the three sprints, and You'll receive a team grade for each of the three sprints based on the technical aspects of the product and team project management."},
    {"question": "What's the minimum final project report grade needed to pass the course?", "expected_answer": "you need to earn a minimum of 75% on your final project report grade."},
    {"question": "Is there any homework besides the final report?", "expected_answer": "Yes, additional homework in project management and development tools."},
    {"question": "How much is the additional homework worth?", "expected_answer": "10% of the final grade."},
    {"question": "What's the minimum credit hour workload estimate?", "expected_answer": "45 hours per credit per term."}
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

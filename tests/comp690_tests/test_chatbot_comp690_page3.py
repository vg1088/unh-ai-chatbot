import pytest
import requests
import re
from sentence_transformers import SentenceTransformer, util

# Load the model for embedding
model = SentenceTransformer('all-MiniLM-L6-v2')

# Define the test cases with expected answers for COMP690 Page 3
test_cases_comp690_page3 = [
    {"question": "What are the three components of the final grade?", "expected_answer": "Class Attendance (10%), Sprint Grade (60%), and Final Project Report (20%). There is also Homework (10%)."},
    {"question": "What percentage of the final grade is class attendance?", "expected_answer": "class attendance is worth 10% of your final grade."},
    {"question": "How is the Sprint Grade calculated?", "expected_answer": "It's calculated as: Teamwork Grade x Sprint Grade."},
    {"question": "What is included in the Teamwork Grade?", "expected_answer": "the Teamwork Grade consists of several components, including participation in group work, communication with team members, collaboration, and the overall contribution to the team's success. It's important to actively engage with your peers and contribute effectively to group assignments and projects."},
    {"question": "What is included in the Sprint Grade?", "expected_answer": " the Sprint Grade includes several components like Weekly progress reports that outline your activities and achievements, Engagement and participation in your internship,Deliverables submitted at the end of each sprint, Feedback from your internship supervisor or mentor."},
    {"question": "What is the percentage weight of the final project report?", "expected_answer": "In COMP690, the final project report is weighted at 30% of your overall grade."},
    {"question": "What are the requirements for the final project report?", "expected_answer": " The requirements for the final project report in COMP690 are outlined in the syllabus. Typically, you'll need to include an overview of your internship experience, the skills you developed, and a reflection on how the experience relates to your academic goals."},
    {"question": "What is the policy on late submissions?", "expected_answer": "The policy is very strict and applies only in exceptional cases (illness, accident, emergencies) with proper documentation."},
    {"question": "Under what circumstances might a late submission be accepted?", "expected_answer": "In COMP690, late submissions are typically accepted under specific circumstances such as documented medical emergencies, family emergencies, or other significant unforeseen events."},
    {"question": "What is the university's policy on attendance?", "expected_answer": "Students are responsible for attending scheduled meetings and abiding by the University Policy on Attendance (as stated in the UNH Student Rights, Rules, and Responsibilities)."},
    {"question": "What should students do if they cannot attend a scheduled meeting?", "expected_answer": "Email the instructor BEFORE the meeting to explain the circumstances and request to be excused. Arrange a meeting to update internship progress."},
    {"question": "What is the minimum number of hours required for the course?", "expected_answer": "A minimum of 45 hours of student academic work per credit per term."},
    {"question": "What should students do if they cannot attend class due to illness?", "expected_answer": "Email the instructor before class, explaining the situation and requesting an excused absence. Follow the late submission policy if applicable."},
    {"question": "What happens if a student fails to comply with late submission rules?", "expected_answer": "In COMP690, if a student fails to comply with late submission rules, it typically results in a penalty, which may include a reduction in the submitted work's grade or no credit being given for that work."},
    {"question": "What are the detailed rubrics for evaluating teamwork?", "expected_answer": "The detailed rubrics are TBA (To Be Announced)."},
    {"question": "What are the detailed rubrics for evaluating sprints?", "expected_answer": "The detailed rubrics are TBA (To Be Announced)."},
    {"question": "What is the minimum grade needed on the final report to pass the course?", "expected_answer": "To pass COMP690, you need to achieve at least a grade of C or higher on your final report."},
    {"question": "Is there any additional homework besides the final report?", "expected_answer": "In COMP690, there are primarily two components you need to complete: the final report and any other assignments specified in the syllabus."},
    {"question": "What is the weight of the homework towards the final grade?", "expected_answer": "In COMP690, homework assignments typically account for 30% of your final grade."},
    {"question": "What is the credit hour workload estimate for this course?", "expected_answer": "A minimum of 45 hours per credit per term."}
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

@pytest.mark.parametrize("test_case", test_cases_comp690_page3)
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

import pytest
import requests
import re
from sentence_transformers import SentenceTransformer, util

# Load the model for embedding
model = SentenceTransformer('all-MiniLM-L6-v2')

# Define the test cases with expected answers for COMP690 Page 5
test_cases_comp690_page5 = [
    {"question": "What questions should be answered in the self-assessment section?", "expected_answer": "The self-assessment should address: What you learned, the relationship of the work to your major, the benefits to you, a comparison of theory (classroom learning) and practice (internship experience), how the project activities correlate with classroom knowledge, how the project will influence future career plans, a reflection on internship experiences (including skills needing development), and advice for a fellow student and/or faculty member."},
    {"question": "What comparison should be made in the self-assessment?", "expected_answer": "In the self-assessment for COMP690, you should compare your internship experiences with the learning objectives outlined in the syllabus. Assess how well you met these objectives, the skills you developed, and the challenges you faced during your internship. This reflection will help you evaluate your growth and the overall value of the experience."},
    {"question": "How should the project's influence on future career plans be discussed?", "expected_answer": "Discuss how the project activities and experiences will influence your future career plans."},
    {"question": "What reflections on the internship experience should be included?", "expected_answer": "In COMP690, your reflections on the internship experience should include thoughts on what you learned, how the experience contributed to your professional development, and any challenges you faced.."},
    {"question": "How long should the self-assessment section be (minimum)?", "expected_answer": "In the COMP690 syllabus, the self-assessment section should be a minimum of 2 pages long."},
    {"question": "What should be included in the conclusion section?", "expected_answer": "In the conclusion section of your COMP690 internship report, you should summarize the key insights and experiences gained during your internship. This includes reflecting on how the internship contributed to your professional development, any challenges you faced and how you overcame them, and the overall impact of the experience on your career goals."},
    {"question": "How long should the conclusion section be?", "expected_answer": "The COMP690 syllabus doesn’t specify an exact length for the conclusion section. However, it’s generally good practice for a conclusion to effectively summarize your main points, reflect on your internship experience, and provide insights or lessons learned."},
    {"question": "What is the required formatting for the internship report (spacing)?", "expected_answer": "For COMP690, the internship report should be formatted with double spacing."},
    {"question": "What is the required page range for the entire report (excluding title page, figures and tables)?", "expected_answer": "Between 6-8 pages."},
    {"question": "What font size is required for the report?", "expected_answer": "The font size required for the report in COMP690 is 12-point."},
    {"question": "What is the breakdown of grading criteria for the final report?", "expected_answer": "60% content, 20% grammar and mechanics, 20% format."},
    {"question": "What is the consequence of failing to meet the page requirements?", "expected_answer": "Up to a 30% deduction from the total report grade."},
    {"question": "What is the minimum percentage needed on the final report to pass the course?", "expected_answer": " The minimum percentage needed on the final report to pass COMP690 is typically 70%. "},
    {"question": "Should bullet points be used in the final report?", "expected_answer": "No, use full sentences."},
    {"question": "How are tables and figures to be handled in the report?", "expected_answer": "All tables and figures must be captioned."},
    {"question": "Can I use a font other than Times New Roman?", "expected_answer": "No, Times New Roman is specified."},
    {"question": "What is the minimum page length for the conclusion section?", "expected_answer": "1 full page."},
    {"question": "What additional skills should be identified for career readiness in the self-assessment?", "expected_answer": " In the COMP690 syllabus, students are encouraged to focus on both technical and soft skills in their self-assessment for career readiness. You may consider identifying skills such as communication, teamwork, problem-solving, adaptability, time management, and any specific technical skills related to your field of study. It's also beneficial to reflect on skills you've gained during your internship experience."},
    {"question": "What advice should be given to a fellow student and/or faculty member?", "expected_answer": "Advice related to the student's internship experience and lessons learned."},
    {"question": "What should be included in a reflection on the internship experience?", "expected_answer": "In a reflection on your internship experience for COMP690, you should include the following elements such as Overview of the Internship, Learning Outcomes, Challenges Faced, Connections to Coursework, Future Application, Personal Growth."},
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

@pytest.mark.parametrize("test_case", test_cases_comp690_page5)
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

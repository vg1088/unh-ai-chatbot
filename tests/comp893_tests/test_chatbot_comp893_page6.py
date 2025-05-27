import pytest
import requests
import re
from sentence_transformers import SentenceTransformer, util

# Load the model for embedding
model = SentenceTransformer('all-MiniLM-L6-v2')


# Define the test cases with expected answers for Page 6
test_cases_page6 = [
    {"question": "Where can I find UNH's academic integrity policy?", "expected_answer": "The syllabus provides a link to the UNH Academic Integrity Policy."},
    {"question": "What's the policy on reporting sexual violence or harassment?", "expected_answer": "If you or someone you know has experienced sexual violence or harassment, it's important to report it to the appropriate university offices. At UNH, faculty are required to report any incidents shared by students to the Title IX Coordinator. You can reach out to Lisa Enright, the Title IX Deputy Intake Coordinator, at lisa.enright@unh.edu or call 603-641-4336."},
    {"question": "How can I report sexual violence or harassment confidentially?", "expected_answer": "Contact the SHARPP Center for Interpersonal Violence Awareness, Prevention, and Advocacy, Civil Rights & Equity Office, UNH Manchester/CPS Title IX Deputy Intake Coordinator, 24 Hour NH Sexual Violence Hotline."},
    {"question": "What confidential support resources are available at UNH Manchester?", "expected_answer": "The SHARPP Extended Services Coordinator, YWCA NH, and the Mental Health Center of Greater Manchester."},
    {"question": "What are the contact details for the UNH Manchester Title IX Deputy Intake Coordinator?", "expected_answer": "you can contact the IX Deputy Intake Coordinator Lisa Enright, Email: lisa.enright@unh.edu, Phone: 603-641-4336, Location: Room 439."},
    {"question": "What library resources are available in Manchester?", "expected_answer": " At UNH Manchester, there are various library resources available to you. The librarians are here to assist with your research needs. You can visit the library’s website for more information on services and to search for reliable academic sources."},
    {"question": "How do I contact the UNH Manchester library?", "expected_answer": "Phone: 603-641-4173 or Email: unhm.library@unh.edu"},
    {"question": "How do I schedule a research appointment with a librarian?", "expected_answer": "The syllabus provides a link to instructions on how to schedule a research appointment."},
    {"question": "How do I use the library's search box?", "expected_answer": "The syllabus provides a link to instructions on how to use the library search box."},
    {"question": "How can I reserve a study room?", "expected_answer": "The syllabus provides a link to instructions on reserving a study room."},
    {"question": "Where can I find resources for citing sources?", "expected_answer": "The syllabus provides a link to resources for citing sources."},
    {"question": "What's the UNH Manchester Library website?", "expected_answer": "UNH Manchester Library."},
    {"question": "Where can I find resources for evaluating sources?", "expected_answer": "The syllabus provides a link to resources for evaluating sources."},
    {"question": "What's the phone number for the SHARPP Center?", "expected_answer": "The phone number for the SHARPP Center for Interpersonal Violence Awareness, Prevention, and Advocacy is (603) 862-7233. If you need TTY services, you can reach them at (800) 735-2964"},
    {"question": "How can I report bias, discrimination, or harassment?", "expected_answer": "Contact UNH's Civil Rights & Equity Office."},
    {"question": "What are the contact details for the UNH Title IX Coordinator?", "expected_answer": "Bo Zaryckyj, Email: Bo.Zaryckyj@unh.edu, Phone: 603-862-2930, If you need to reach out to the Title IX Deputy Intake Coordinator at UNH Manchester, you can contact Lisa Enright at Email: lisa.enright@unh.edu, Phone: 603-641-4336."},
    {"question": "What app provides access to reporting options and resources?", "expected_answer": "The app that provides access to reporting options and resources is the uSafeUS app. It helps keep the reporting options and resources easily accessible for students on their phones."},
    {"question": "Where can I find the UNH Academic Integrity Policy?", "expected_answer": "You can find the UNH Academic Integrity Policy by following this link: [Academic Integrity Policy link](https://cps.unh.edu/library)"},
    {"question": "Where can I find more information about reporting procedures?", "expected_answer": "For more information about reporting procedures related to Title IX at UNH, including your rights and available options, you can visit the university's Title IX resources. They provide details on what happens when you report, how your information is handled, and options for anonymous reporting. You can usually find this information on the university's official website or by contacting the Title IX Coordinator directly."},
    {"question": "What's the number for the 24-hour NH Domestic Violence Hotline?", "expected_answer": "The number for the 24-hour NH Domestic Violence Hotline is 1-866-644-3574"}
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

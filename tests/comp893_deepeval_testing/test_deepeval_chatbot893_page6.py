import pytest
import requests
import os
import configparser
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase

# Load the API key from config.txt and set it as an environment variable
config = configparser.ConfigParser()
config.read("config.txt")

try:
    os.environ["OPENAI_API_KEY"] = config.get("settings", "openai_key")
except (configparser.NoSectionError, configparser.NoOptionError) as e:
    raise ValueError("OpenAI API key not found in config.txt under [settings] section.") from e

# Define the test cases with expected answers for Page 6
test_cases = [
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

# Helper function to get chatbot response
def get_chatbot_response(session, question):
    try:
        response = session.post("http://localhost:1896/llm_response", data={"message": question})
        response.raise_for_status()
        return response.text.strip()
    except requests.exceptions.RequestException as e:
        print(f"Error connecting to chatbot API: {e}")
        return None

# Explicitly set the context to COMP893
def set_course_context(session):
    context_question = "I am asking about COMP893"
    response = get_chatbot_response(session, context_question)
    if response is None:
        print("Failed to set course context.")

@pytest.mark.parametrize("test_case", test_cases)
def test_chatbot_responses(test_case):
    session = requests.Session()
    set_course_context(session)
    
    question = test_case["question"]
    expected_answer = test_case["expected_answer"]
    
    # Get chatbot response
    actual_response = get_chatbot_response(session, question)
    if actual_response is None:
        pytest.fail(f"Chatbot did not return a response for question: {question}")
    
    # Function to clean chatbot responses
    def clean_response(response):
        return response.split(" If you have any more questions")[0].strip()

    # Clean the responses
    cleaned_actual_response = clean_response(actual_response)
    cleaned_expected_answer = clean_response(expected_answer)
    
    # Initialize the AnswerRelevancyMetric
    relevancy_metric = AnswerRelevancyMetric(
        threshold=0.5,
        model="gpt-4",
        include_reason=True,
        async_mode=False  # Disable asynchronous execution
    )
    
    # Create the test case for relevancy
    relevancy_test_case = LLMTestCase(
        input=question,
        actual_output=cleaned_actual_response
    )
    
    # Evaluate relevancy
    relevancy_metric.measure(relevancy_test_case)
    relevance_score = relevancy_metric.score
    relevance_reason = relevancy_metric.reason
    
    # Log the result
    print(f"Question: {question}")
    print(f"Actual Response: {actual_response}")
    print(f"Cleaned Response: {cleaned_actual_response}")
    print(f"Relevance Score: {relevance_score}")
    print(f"Reason: {relevance_reason}")
    
    # Adjusted threshold
    assert relevance_score >= 0.5, f"Failed relevance for question: {question}, Score: {relevance_score}, Reason: {relevance_reason}"

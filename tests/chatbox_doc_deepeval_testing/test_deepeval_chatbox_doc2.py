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

# Define the test cases with expected answers for chatbox_doc Page 2
test_cases = [
    {"question": "How many credits is COMP 892 worth?", "expected_answer": "1-3 variable credits."},
    {"question": "Is COMP 892 only for full-time workers?", "expected_answer": "COMP892 is designed for students who are currently employed in a field related to their studies. It is intended for those who can apply their work experience to their academic learning. If you're not currently working full-time, you might want to consider other internship options."},
    {"question": "When is COMP 892 offered?", "expected_answer": "Fall, spring, and summer semesters."},
    {"question": "What is COMP 893 for?", "expected_answer": " COMP893, or the Team Project Internship, is a course that allows students to work collaboratively on a team project within an internship setting. The course emphasizes teamwork, project management, and practical applications of computing systems. Students engage in IT activities, identify user needs, and analyze how to develop and operate these systems effectively."},
    {"question": "What are the requirements for COMP 893?", "expected_answer": "Being in the final semester and needing to fulfill internship requirements."},
    {"question": "How do I register for internship courses?", "expected_answer": "You need instructor permission; email the faculty internship coordinator or course instructor."},
    {"question": "Do I need permission to register for internship courses?", "expected_answer": "Yes, you need permission to register for internship courses."},
    {"question": "How do I sign up for COMP690?", "expected_answer": "You need to email the faculty internship coordinator or course instructor for permission to register."},
    {"question": "Where are the best places to look for internships?", "expected_answer": "Handshake:university's career services platform where employers post internships specifically for students, company websites:Check the careers section of companies you're interested in, as many post internships directly on their site, Websites like Indeed, LinkedIn, Glassdoor, and SimplyHired often list internship opportunities."},
    {"question": "What is Handshake?", "expected_answer": "Handshake is a career services platform that connects students with job and internship opportunities. It allows students to search for positions, research companies, and apply for internships and jobs. Additionally, it provides resources for resume building, interview preparation, and networking with employers. It's a valuable tool for students looking to enhance their internship and job search experience."},
    {"question": "Are there job sites besides Handshake?", "expected_answer": "Yes, there are several job sites besides Handshake where you can search for internships and job opportunities. Some popular options include:LinkedIn, Indeed, Glassdoor, SimplyHired,Internships.com. These platforms offer a variety of listings and can help you find opportunities in your field of interest."},
    {"question": "What career services are available?", "expected_answer": "Resume writing, interview coaching, and other career advice."},
    {"question": "Does the career office help with cover letters?", "expected_answer": "Yes, the career office typically provides resources and assistance for writing cover letters. They may offer workshops, one-on-one appointments, and reviews of your cover letter drafts."},
    {"question": "If I'm currently working, do I still need to take an internship course?", "expected_answer": "Yes, you still need to take an internship course, even if you are currently employed in the field. However, you may have the option to use the applied research option to fulfill the internship requirement."},
    {"question": "Do I still need an internship course if I already have a job?", "expected_answer": " Yes, you are still required to complete an internship course even if you currently have a job in the field. However, you have the option to use the applied research option to fulfill the internship requirement."},
    {"question": "Do I have to get a new job for the internship course?", "expected_answer": "No, you do not have to get a new job for the internship course. If you are currently employed in the field, you can use your existing job to fulfill the internship requirement. Just be sure to meet the necessary hours and requirements outlined in your course syllabus."},
    {"question": "I already work; which internship course is best for me?", "expected_answer": "If you already work in a relevant field, you might consider the applied research option to fulfill the internship requirement. Both COMP690 and COMP893 require you to complete internship work, but COMP690 is typically more focused on practical experience, while COMP893 may involve a research component. It would be best to discuss your specific situation with the faculty internship coordinator to determine which course aligns best with your current work experience."},
    {"question": "What if I work part-time?", "expected_answer": " If you work part-time, you can still fulfill your internship requirement by using that experience. Just make sure it meets the necessary criteria for your course."},
    {"question": "What if I already have a tech job?", "expected_answer": "If you already have a tech job, you can still fulfill the internship requirement. You may use the applied research option to meet the internship criteria."},
    {"question": "I have a full-time tech job; what's my best option?", "expected_answer": "If you already work full-time in the tech field, you can consider using the applied research option to fulfill your internship requirement. This allows you to leverage your current job experience towards your internship credits while engaging in meaningful projects at your workplace."}
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
    context_question = "I am asking about general internship questions"
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

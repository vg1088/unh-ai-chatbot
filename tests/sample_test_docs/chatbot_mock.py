# chatbot_mock.py

import re

def get_response(question):
    # A simple dictionary with hardcoded responses for testing
    responses = {
        "What time is the M2 section?": "The M2 section is at 1:10 PM.",
        "Who is the instructor?": "The instructor is Professor Karen Jin.",
        "How many credits is this course?": "1-3 credits.",
        "Where is the class held?": "The class is held in room P142.",
        "What time is the M1 section on Wednesdays?": "9:10 AM to 12:00 PM.",
        "What time is the M2 section on Wednesdays?": "1:10 PM to 4:00 PM.",
        "What's the instructor's title?": "Associate Professor.",
        "What department is the professor in?": "Department of Applied Engineering and Sciences.",
        "Where is the professor's office?": "Rm 139, Pandora Mill building.",
        "What is the professor's Zoom link?": "Join our Cloud HD Video Meeting.",
        "What is the professor's email?": "karen.jin@unh.edu.",
        "What are the professor's office hours?": "Monday 1:00 PM to 4:00 PM and Friday 9:00 AM to 12:00 PM.",
        "Are office hours available online?": "Yes, over Zoom.",
        "How do I schedule an appointment with the professor?": "Email the professor to make an appointment.",
        "What building is the class in?": "The syllabus does not specify the building.",
        "What's the course description?": "The course provides experiential learning through team-based projects.",
        "What are the student learning outcomes?": "Analyzing computing problems, designing solutions, working in teams.",
        "Is there a phone number for the professor?": "The syllabus does not list a phone number.",
        "What is the professor's full title?": "Associate Professor, Department of Applied Engineering and Science.",
        "What happens in week 1?": "Class introduction, team setup, project management intro, scrum workflow intro, and project goal.",
        "When is the project kickoff?": "Week 2 (September 4th).",
        "When does the first sprint begin?": "Week 4 (September 18th).",
        "How often are scrum meetings during the first sprint?": "Monday, Wednesday, and Friday.",
        "When does the first sprint end?": "Week 6 (October 2nd).",
        "What's involved in a sprint retrospective?": "Reviewing the sprint and identifying areas for improvement.",
        "When is the second sprint planning meeting?": "Week 7 (October 9th).",
        "When does the second sprint start?": "Week 7 (October 9th).",
        "Does the scrum meeting schedule change during the second sprint?": "Yes, the frequency changes throughout the course.",
        "When is Thanksgiving Break?": "Week 13 (November 20th).",
        "What's the focus of week 3?": "Environment setup (Jira), creating the project backlog, user stories, tasks, and bugs.",
        "When are scrum meetings held only on Mondays?": "During Week 6 (10/2) and Week 11 (11/6).",
        "What happens during week 10?": "Scrum meetings (Monday, Wednesday, Friday).",
        "When is the sprint review for the first sprint?": "Week 6 (October 2nd).",
        "What are the activities for Week 9?": "Scrum meetings (Monday, Wednesday, Friday).",
        "When does the third sprint start?": "Week 12 (November 13th).",
        "How often are scrum meetings in week 13?": "Monday, Wednesday, and Friday; plus a weekly status report.",
        "What is the project goal (referencing information from page 1)?": "Hands-on experience and contribution to real-world IT products/services.",
        "What tools are used for project management?": "Jira is mentioned in the syllabus.",
        "What is the purpose of sprint planning?": "To create a plan for the upcoming sprint based on the Product Backlog.",
        "What should be included in my self-assessment?": "What you learned, project's relevance, benefits gained, comparison of theory and practice.",
        "What's the policy on late submissions?": "Very strict; only granted in exceptional, documented cases of illness, accident, or emergency.",
        "Where can I find UNH's academic integrity policy?": "The syllabus provides a link to the UNH Academic Integrity Policy.",
        "How do I contact the UNH Manchester library?": "Phone: 603-641-4173 or Email: unhm.library@unh.edu.",
        "What's the phone number for the SHARPP Center?": "(603) 862-7233/TTY (800) 735-2964.",
        "How can I report bias, discrimination, or harassment?": "Contact UNH's Civil Rights & Equity Office."
    }

    # Default response if the question is not found in predefined responses
    return responses.get(question, "I'm here to help, but I didn't understand your question.")

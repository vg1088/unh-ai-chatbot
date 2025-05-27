import os
import time
import csv
import PyPDF2
import configparser
from qdrant_client import QdrantClient
from qdrant import qdrantsearch
from fastembed import TextEmbedding
from flask import Flask, render_template, request, redirect, session, make_response
from openai import OpenAI
from itertools import chain
from datetime import datetime 

config = configparser.ConfigParser()
config.read("config.txt")

app = Flask(__name__)
app.secret_key = "comp690"

@app.route("/")
def hello_world():
    session['history'] = [prompt]
    session['course'] = ""
    return render_template("chatbotUI.html")

@app.route('/llm_response', methods=['POST', 'PUT'])
def handle_post():
    if request.method == 'POST':
        if 'history' not in session:
            session['history'] = [prompt]
            session['course'] = ""

        message = request.form['message']
        session['history'].append( {"role": "user", "content": f"{message}"})

        response = make_response(get_response(session))
        response.mimetype = "text/plain"
        session['history'].append( {"role": "assistant", "content": f"{response}"})
        return response
    elif request.method == 'PUT':
        session['history'] = [prompt]
        session['course'] = ""
        response = "all good"
        return response

def answer_question(messages, chunks):
    history = messages.copy()
    loc_chunks = chunks.copy()
    loc_chunks = [mes.payload.values() for x in chunks for mes in x]    
    
    for payload in loc_chunks:
        history.append({"role": "system", "content": f"""context:
        {payload}
        """})
       # print(payload)

    response = open_client.chat.completions.create(model = "gpt-4o-mini", messages = history)
    return response

def main():
    global qdrant_client
    global openai_key
    global open_client
    global embed_model
    global prompt
    global data_dir
    data_dir=config.get("settings", "data_dir")

    prompt = {"role": "system", "content": f"""
You are a friendly, knowledgeable chatbot designed to assist students with,questions about their internship experience,
based on course syllabi and internship-related FAQs.
You can refer to documents such as the "COMP690 Internship Experience" syllabus, the "COMP893 Internship Experience" syllabus, and the "Chatbox.pdf" document for general internship FAQs.
If the question is course-specific (e.g., office hours, class schedule), refer to the appropriate syllabus (COMP690 or COMP893).
For more general internship-related questions (e.g., internship hours,CPT, or Handshake),
refer to the information in "Chatbox.pdf." When a user says "thank you," respond with "You're welcome! If you have any more questions, feel free to ask!" instead of repeating your previous response.
Follow these guidelines to ensure accurate and natural responses:
1. Determine the Context:
Identify which course (COMP690 or COMP893) or general topic the user is asking about. If it is unclear, politely ask for clarification (e.g., "Are you asking about COMP690, COMP893, or a general internship question?").
2. Prioritize the Relevant Document:
If the question is course-specific (e.g., office hours, class schedule), refer to the appropriate syllabus (COMP690 or COMP893).
For general internship-related questions (e.g., internship hours, CPT, or Handshake), refer to the "Chatbox.pdf."
3. Provide Clear, Direct Answers:
         Respond briefly and directly to questions like "How many credits?" or "Where is the class?" using the appropriate document.
         Avoid unnecessary details unless the user asks for more information. (e.g., "How many credits?" → "4 credits").
         Use the following preferred formatting:
         Example:
         Credits: 4 credits
         Requirements: Complete 150 hours of internship work
         Eligibility: Required even if you are currently employed in the field do not format responses like this,
         Example of what not to do:
         - **Credits**:
         - **Requirements**:
         - **Eligibility**:
         - **Course Components**:
         - **Registration**:
4. Enhance Conversational Tone:
Avoid robotic phrasing like starting with "Answer:". Instead, simply respond with the relevant information in a natural, friendly manner, as if speaking to a student in person.
5. Handle FAQs Efficiently:
For general internship FAQs (e.g., registering internships, Handshake), rely on "Chatbox.pdf" as your main source of information.
6. Ask for Clarification When Needed:
If the query is unclear or applies to multiple contexts (e.g., a question about hours), politely ask for clarification before providing a response.
7. Address Missing Information Gracefully:
If the requested information is not available in the provided documents, reply with: "I don’t have that information right now," or offer a suggestion (e.g., check with the instructor or syllabus for updates).
8. Avoid Irrelevant Details:
Stay focused on the specific question asked. For example, if the user asks about credits, don’t dive into workload unless it is relevant.
9. Be Flexible with Wording Variations:
Recognize and interpret common misspellings or different phrasings, responding to the user’s intended meaning.
10. Maintain a Friendly, Natural Tone:
Ensure your responses feel like a conversation with a professor or teaching assistant—approachable, professional, and helpful.
""" }
    openai_key = config.get("settings", "openai_key")
    open_client = OpenAI(api_key = openai_key)

    qdrant_client = QdrantClient(host=config.get("settings", "qdrant_host"), port=6333)
    embed_model = TextEmbedding()

def get_response(session):
    messages = session['history']
    question = messages[-1]['content']

    t_in = time.time()
    #print(messages)
    if session['course'] == "":
        chunks = get_context(session, question)
    chunks = get_rag(session, question)

    answer = answer_question(messages, chunks)
    response = answer.choices[0].message.content
    t_fin = time.time()

    resp_time = t_fin - t_in
    chunks = [mes.payload.values() for x in chunks for mes in x]    

    if session['course']:
        fields=[question, chunks, answer, resp_time, session['course'], datetime.now()]
    else:
            fields=[question, chunks, answer, resp_time, "none", datetime.now()]
    with open(data_dir + 'log.csv', 'a+', newline='', encoding="utf-8") as log:
        writer = csv.writer(log)
        writer.writerow(fields)
    return response

def get_rag(session, question):
    chunks = []
    if session['course'] == "":
        chunks.append(qdrantsearch.search_db(qdrant_client, question, embed_model, "default"))
    else:
        chunks.append(qdrantsearch.search_db(qdrant_client, question, embed_model, "default"))
        print(session['course'])
        chunks.append(qdrantsearch.search_db(qdrant_client, question, embed_model, session['course']))
    return chunks



def get_context(messages, question):
    messages = session["history"].copy()
    messages[0] = {"role": "system", "content": f"""
    You are a bot that categorizes questions. Given the ENTIRE chat history from the user,
    determine if they are asking about Comp 893, or Comp 690. 
                   
    Respond with only "Comp 893", or "Comp 690". If you are unsure, respond with "not sure"

    """
    }
    response = open_client.chat.completions.create(model = "gpt-4o-mini", messages = messages)
    course = response.choices[0].message.content

    match course:
        case "Comp 893":
            #chunks = qdrantsearch.search_db(qdrant_client, question, embed_model, "893")
            session['course'] = 893
        case "Comp 690":
            #chunks = qdrantsearch.search_db(qdrant_client, question, embed_model, "690")
            session['course'] = 690
        case _:
            #chunks = qdrantsearch.search_db(qdrant_client, question, embed_model, "default")
            session['course'] = ""

    fields=[question, course, session['course']]
    with open(data_dir + 'raglog.csv', 'a+', newline='', encoding="utf-8") as log:
        writer = csv.writer(log)
        writer.writerow(fields)
    return session['course']

if __name__ == "__main__":
    main()
    app.run(host=config.get("settings", "bot_ip"), port = config.get("settings", "bot_port"))

import os
import logging
import time
import json
import sqlite3
import numpy as np
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field
from typing import List, Dict, TypedDict
import chardet
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import base64
from email.mime.text import MIMEText
import pandas as pd
import pdfplumber
import ollama
from IPython.display import Image, display
from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod, NodeStyles

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('job_screening.log'),  # Log to file
        logging.StreamHandler()                    # Log to console
    ]
)
logger = logging.getLogger(__name__)

# Gmail API scopes
SCOPES = ['https://www.googleapis.com/auth/gmail.send', 'https://www.googleapis.com/auth/gmail.readonly']

# Gmail API authentication function
def get_gmail_service():
    creds = None
    if os.path.exists('token.json'):
        try:
            creds = Credentials.from_authorized_user_file('token.json', SCOPES)
        except Exception as e:
            logger.error(f"Failed to load credentials from token.json: {e}")
            return None
            
    if not creds or not creds.valid:
        try:
            flow = InstalledAppFlow.from_client_secrets_file('desktop.json', SCOPES)
            creds = flow.run_local_server(port=8080)
            with open('token.json', 'w') as token:
                token.write(creds.to_json())
        except Exception as e:
            logger.error(f"Failed to authenticate Gmail API: {e}")
            return None
    
    try:
        service = build('gmail', 'v1', credentials=creds)
        return service
    except Exception as e:
        logger.error(f"Failed to build Gmail service: {e}")
        return None

# Email sending function with detailed logging
def send_email(service, to, subject, message_text):
    if service is None:
        logger.warning("Gmail service not available, cannot send email")
        return None
    message = MIMEText(message_text)
    message['to'] = to
    message['from'] = 'me'
    message['subject'] = subject
    raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
    body = {'raw': raw}
    try:
        message = service.users().messages().send(userId='me', body=body).execute()
        logger.info(f"Email sent successfully to {to} with subject: {subject}")
        return message['id']
    except HttpError as error:
        logger.error(f"HTTP error sending email to {to}: {error}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error sending email to {to}: {e}")
        return None

# Function to get email messages
def get_latest_reply(service, to_email):
    try:
        query = f"from:{to_email} after:{int(time.time()) - 3600}"  # Last hour
        messages = service.users().messages().list(userId='me', q=query).execute()
        if 'messages' in messages and messages['messages']:
            message = service.users().messages().get(userId='me', id=messages['messages'][0]['id'], format='full').execute()
            payload = message['payload']
            parts = payload.get('parts', [payload])
            for part in parts:
                if part.get('mimeType') == 'text/plain':
                    data = part['body']['data']
                    reply = base64.urlsafe_b64decode(data).decode('utf-8')
                    logger.info(f"Received reply from {to_email}: {reply}")
                    return reply
            return None
        return None
    except HttpError as error:
        logger.error(f"HTTP error fetching reply from {to_email}: {error}")
        return None

# Define Pydantic models
class JobDescription(BaseModel):
    education: str = Field(description="Required education level or degree")
    skills: List[str] = Field(description="Required technical skills")
    tools: List[str] = Field(description="Specific tools or software required")
    experience_years: str = Field(description="Required years of experience")
    role: str = Field(description="Primary job role or title")
    projects: List[str] = Field(description="Types of projects or tasks required")
    responsibilities: List[str] = Field(description="Job responsibilities")
    certifications: List[str] = Field(description="Required certifications")

class CVProfile(BaseModel):
    candidate_name: str = Field(description="Name of the candidate")
    email: str = Field(description="Candidate's email address")
    education: str = Field(description="Candidate's education level or degree")
    skills: List[str] = Field(description="Candidate's technical skills")
    tools: List[str] = Field(description="Tools or software used by candidate")
    experience_years: str = Field(description="Total years of experience")
    role: str = Field(description="Primary role or title from experience")
    projects: List[str] = Field(description="Projects or tasks completed")
    responsibilities: List[str] = Field(description="Responsibilities from experience")
    certifications: List[str] = Field(description="Candidate's certifications")

# Define state for LangGraph
class JobScreeningState(TypedDict):
    job_description: Dict
    cv_data: Dict
    match_score: float
    shortlisted: bool
    interview_request: str
    email_id: str
    initial_slot: str
    rescheduled: bool

# Initialize SQLite database
def init_db():
    conn = sqlite3.connect('job_screening4.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS job_data
                 (id INTEGER PRIMARY KEY, type TEXT, data TEXT)''')
    conn.commit()
    return conn

# Store data in SQLite
def store_data(conn, data_type: str, data: Dict):
    c = conn.cursor()
    c.execute("INSERT INTO job_data (type, data) VALUES (?, ?)", (data_type, json.dumps(data)))
    conn.commit()

# Generate embeddings
def generate_embedding(text: str) -> np.ndarray:
    try:
        if not text or text.strip() == "":
            return np.zeros(768)
        response = ollama.embeddings(model="nomic-embed-text", prompt=text)
        if 'embedding' in response and isinstance(response['embedding'], list):
            embedding = np.array(response['embedding'])
            if embedding.size == 0:
                return np.zeros(768)
            elif embedding.shape[0] != 768:
                return np.zeros(768)
            return embedding
        else:
            return np.zeros(768)
    except Exception as e:
        logger.error(f"Error generating embedding for text '{text}': {e}")
        return np.zeros(768)

# Agent: Job Description Summarizer
def summarize_job_description(state: JobScreeningState) -> JobScreeningState:
    job_desc_text = state.get('job_description', {}).get('text', "No job description provided")
    logger.info("Summarizing job description")
    prompt = (
        f"Extract and format the following information from the job description in a concise, relevant manner:\n"
        f"- Education: A single string of the required degree or level (e.g., 'Bachelor’s in Computer Science').\n"
        f"- Skills: A list of specific technical skills (e.g., 'Python', 'SQL'). Infer related skills (e.g., 'Machine Learning' implies 'Python').\n"
        f"- Tools: A list of specific tools or software (e.g., 'Tableau', 'Power BI') mentioned in the text.\n"
        f"- Experience_years: A single string of required years (e.g., '3+ years').\n"
        f"- Role: A single string of the primary job role (e.g., 'Data Scientist').\n"
        f"- Projects: A list of project types or tasks (e.g., 'Develop ML models', 'Build dashboards').\n"
        f"- Responsibilities: A list of specific duties (e.g., 'Analyze datasets', 'Deploy models').\n"
        f"- Certifications: A list of required certifications (e.g., 'AWS Certified').\n"
        f"Provide the values directly based on the text, with no additional explanations:\n{job_desc_text}"
    )
    response = ollama.generate(
        model="llama3.2:1b",
        prompt=prompt,
        format=JobDescription.model_json_schema(),
        options={"temperature": 0}
    )
    if hasattr(response, 'response'):
        jd_data = json.loads(response.response)
    else:
        jd_data = {"error": "Failed to parse job description"}
        logger.error("Failed to parse job description")
    state['job_description'] = jd_data
    conn = init_db()
    store_data(conn, "job_description", jd_data)
    conn.close()
    return state

# Agent: Recruiting Agent (CV Extraction and Matching)
def process_cv_and_match(state: JobScreeningState) -> JobScreeningState:
    cv_text = state.get('cv_data', {}).get('text', "No CV provided")
    logger.info(f"Processing CV: {state.get('cv_data', {}).get('file', 'Unknown CV')}")
    prompt = (
        f"Extract and format the following information from the CV text in a concise, relevant manner:\n"
        f"- Candidate_name: A single string of the candidate’s full name.\n"
        f"- Email: A single string of the candidate’s email address (e.g., 'john.doe@example.com'). Look for terms like 'email', 'contact', or standard email formats.\n"
        f"- Education: A single string of the highest degree (e.g., 'Ph.D. in Artificial Intelligence').\n"
        f"- Skills: A list of specific technical skills (e.g., 'Python', 'SQL'). Infer related skills (e.g., 'Machine Learning' implies 'Python', 'Data Scientist' implies 'SQL', 'Tableau').\n"
        f"- Tools: A list of specific tools or software (e.g., 'Tableau', 'AWS') mentioned or inferred from experience (e.g., 'Data Scientist' implies 'Tableau').\n"
        f"- Experience_years: A single string of total years of experience (e.g., '5 years'), calculated from dates if provided.\n"
        f"- Role: A single string of the primary role from experience (e.g., 'Data Scientist').\n"
        f"- Projects: A list of project types or tasks (e.g., 'Built ML models', 'Created dashboards') inferred from experience.\n"
        f"- Responsibilities: A list of specific duties (e.g., 'Analyzed data', 'Deployed models') from experience.\n"
        f"- Certifications: A list of specific certifications (e.g., 'AWS Certified').\n"
        f"Provide the values directly based on the text, with no additional explanations. If no email is found, return 'email_not_found':\n{cv_text}"
    )
    response = ollama.generate(
        model="llama3.2:1b",
        prompt=prompt,
        format=CVProfile.model_json_schema(),
        options={"temperature": 0}
    )
    if hasattr(response, 'response'):
        cv_data = json.loads(response.response)
    else:
        cv_data = {"error": "Failed to parse CV"}
        logger.error(f"Failed to parse CV: {cv_text[:50]}...")
    
    if cv_data.get('email') == 'email_not_found':
        logger.warning(f"No email found in CV: {cv_text[:50]}...")
        raise ValueError("No email address found in CV. Please ensure the CV contains a valid email.")
    
    state['cv_data'] = cv_data
    conn = init_db()
    store_data(conn, "cv_data", cv_data)

    jd = state['job_description']
    cv = state['cv_data']

    jd_education_emb = generate_embedding(jd.get('education', ''))
    cv_education_emb = generate_embedding(cv.get('education', ''))
    jd_skills_emb = generate_embedding(' '.join(jd.get('skills', [])))
    cv_skills_emb = generate_embedding(' '.join(cv.get('skills', [])))
    jd_tools_emb = generate_embedding(' '.join(jd.get('tools', [])))
    cv_tools_emb = generate_embedding(' '.join(cv.get('tools', [])))
    jd_exp_years_emb = generate_embedding(jd.get('experience_years', ''))
    cv_exp_years_emb = generate_embedding(cv.get('experience_years', ''))
    jd_role_emb = generate_embedding(jd.get('role', ''))
    cv_role_emb = generate_embedding(cv.get('role', ''))
    jd_projects_emb = generate_embedding(' '.join(jd.get('projects', [])))
    cv_projects_emb = generate_embedding(' '.join(cv.get('projects', [])))
    jd_resp_emb = generate_embedding(' '.join(jd.get('responsibilities', [])))
    cv_resp_emb = generate_embedding(' '.join(cv.get('responsibilities', [])))
    jd_certs_emb = generate_embedding(' '.join(jd.get('certifications', [])))
    cv_certs_emb = generate_embedding(' '.join(cv.get('certifications', [])))

    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    education_sim = cosine_similarity(jd_education_emb, cv_education_emb)
    skills_sim = cosine_similarity(jd_skills_emb, cv_skills_emb)
    tools_sim = cosine_similarity(jd_tools_emb, cv_tools_emb)
    exp_years_sim = cosine_similarity(jd_exp_years_emb, cv_exp_years_emb)
    role_sim = cosine_similarity(jd_role_emb, cv_role_emb)
    projects_sim = cosine_similarity(jd_projects_emb, cv_projects_emb)
    resp_sim = cosine_similarity(jd_resp_emb, cv_resp_emb)
    certs_sim = cosine_similarity(jd_certs_emb, cv_certs_emb)

    match_score = (
        (0.15 * education_sim) +
        (0.20 * skills_sim) +
        (0.15 * tools_sim) +
        (0.10 * exp_years_sim) +
        (0.15 * role_sim) +
        (0.10 * projects_sim) +
        (0.10 * resp_sim) +
        (0.05 * certs_sim)
    ) * 100

    state['match_score'] = min(match_score, 100) / 100
    logger.info(f"Match score calculated: {state['match_score']}")
    store_data(conn, "match_score", {"score": state['match_score']})
    conn.close()
    return state

# Agent: Shortlisting Agent
def shortlist_candidates(state: JobScreeningState) -> JobScreeningState:
    threshold = 0.70
    state['shortlisted'] = bool(state['match_score'] >= threshold)
    logger.info(f"Candidate shortlisted: {state['shortlisted']} (Score: {state['match_score']}, Threshold: {threshold})")
    conn = init_db()
    store_data(conn, "shortlisted", {"status": state['shortlisted']})
    conn.close()
    return state

# Agent: Scheduling Agent
def schedule_interview_agent(state: JobScreeningState) -> JobScreeningState:
    service = get_gmail_service()
    conn = init_db()
    default_slot = "2025-04-15, 10:00 AM, Virtual"

    if not state['shortlisted']:
        state['interview_request'] = "Candidate not shortlisted."
        logger.info(f"Candidate not shortlisted, no interview scheduled")
        store_data(conn, "interview_request", {"request": state['interview_request']})
        conn.close()
        return state

    candidate_name = state['cv_data'].get('candidate_name', 'Candidate')
    to_email = state['cv_data'].get('email')
    logger.info(f"Scheduling interview for {candidate_name} (Email: {to_email})")
    if not to_email:
        state['interview_request'] = "Failed to send interview request: No email found."
        logger.warning(f"No email found for {candidate_name}")
        store_data(conn, "interview_request", {"request": state['interview_request']})
        conn.close()
        return state

    subject = f"Interview Request for {candidate_name}"
    message_text = (
        f"Dear {candidate_name},\n\n"
        f"You have been shortlisted for an interview. We propose the following slot:\n"
        f"Slot: [Date: {default_slot}, Format: Virtual]\n\n"
        f"Please reply to this email with 'yes' to confirm or 'no, suggest [date time]' (e.g., 'no, suggest 2025-04-16 2:00 PM') to reschedule.\n\n"
        f"Best regards,\nRecruitment Team"
    )

    email_id = send_email(service, to_email, subject, message_text)
    if not email_id:
        state['interview_request'] = "Failed to send initial interview request."
        logger.error(f"Failed to send interview request to {to_email}")
        store_data(conn, "interview_request", {"request": state['interview_request']})
        conn.close()
        return state

    state['email_id'] = email_id
    state['interview_request'] = message_text
    state['initial_slot'] = default_slot
    state['rescheduled'] = False

    max_polls = 2
    poll_interval = 5

    for _ in range(max_polls):
        reply = get_latest_reply(service, to_email)
        if reply:
            if "yes" in reply.lower():
                state['interview_request'] = f"{candidate_name} confirmed availability for {default_slot}"
                logger.info(f"{candidate_name} confirmed slot: {default_slot}")
                break
            elif "no" in reply.lower() and "suggest" in reply.lower():
                suggested_slot = reply.lower().split("suggest")[-1].strip()
                subject = f"Rescheduled Interview for {candidate_name}"
                message_text = (
                    f"Dear {candidate_name},\n\n"
                    f"Thank you for your response. We have rescheduled your interview to:\n"
                    f"Slot: [Date: {suggested_slot}, Format: Virtual]\n\n"
                    f"Please reply with 'yes' to confirm.\n\n"
                    f"Best regards,\nRecruitment Team"
                )
                email_id = send_email(service, to_email, subject, message_text)
                if email_id:
                    state['email_id'] = email_id
                    state['interview_request'] = message_text
                    state['initial_slot'] = suggested_slot
                    state['rescheduled'] = True
                    for _ in range(max_polls):
                        reply = get_latest_reply(service, to_email)
                        if reply and "yes" in reply.lower():
                            state['interview_request'] = f"{candidate_name} confirmed rescheduled slot: {suggested_slot}"
                            logger.info(f"{candidate_name} confirmed rescheduled slot: {suggested_slot}")
                            break
                        time.sleep(poll_interval)
                    break
                else:
                    state['interview_request'] = "Failed to send rescheduled interview request."
                    logger.error(f"Failed to send rescheduled request to {to_email}")
                    break
            else:
                state['interview_request'] = f"Invalid response from candidate: {reply}"
                logger.warning(f"Invalid response from {to_email}: {reply}")
                break
        time.sleep(poll_interval)

    if not state['interview_request'].startswith("Candidate confirmed"):
        state['interview_request'] += " No valid response received within polling period."
        logger.info(f"No valid response received from {to_email}")

    store_data(conn, "interview_request", {"request": state['interview_request']})
    conn.close()
    return state

# Build the LangGraph workflow
graph_builder = StateGraph(JobScreeningState)
graph_builder.add_node("summarize_job", summarize_job_description)
graph_builder.add_node("process_cv", process_cv_and_match)
graph_builder.add_node("shortlist", shortlist_candidates)
graph_builder.add_node("schedule", schedule_interview_agent)
graph_builder.add_edge(START, "summarize_job")
graph_builder.add_edge("summarize_job", "process_cv")
graph_builder.add_edge("process_cv", "shortlist")
graph_builder.add_edge("shortlist", "schedule")
graph_builder.add_edge("schedule", END)
graph = graph_builder.compile()

# Read data and process CVs
try:
    with open("job_description.csv", 'rb') as f:
        result = chardet.detect(f.read())
    df = pd.read_csv("job_description.csv", encoding=result['encoding'])
    job_description = df['Job Description'][0]
    logger.info("Job description loaded successfully")
except Exception as e:
    job_description = "Error reading job description"
    logger.error(f"Failed to read job_description.csv: {e}")

cv_folder = "./test"
cv_results = []

for cv_file in os.listdir(cv_folder):
    if cv_file.lower().endswith('.pdf'):
        cv_path = os.path.join(cv_folder, cv_file)
        logger.info(f"Starting processing for CV: {cv_file}")
        try:
            with pdfplumber.open(cv_path) as pdf:
                cv_text = ''.join(page.extract_text() for page in pdf.pages if page.extract_text())
        except Exception as e:
            cv_text = f"Error reading CV: {cv_file}"
            logger.error(f"Failed to read CV {cv_file}: {e}")

        initial_state = {
            "job_description": {"text": job_description},
            "cv_data": {"text": cv_text, "file": cv_file},  # Added file name for logging
            "match_score": 0.0,
            "shortlisted": False,
            "interview_request": "",
            "email_id": "",
            "initial_slot": "2025-04-15, 10:00 AM, Virtual",
            "rescheduled": False
        }

        try:
            result = graph.invoke(initial_state)
            cv_results.append({
                "cv_file": cv_file,
                "job_description": result['job_description'],
                "cv": result['cv_data'],
                "match_score": result['match_score'],
                "shortlisted": result['shortlisted'],
                "interview_request": result['interview_request'],
                "email_id": result['email_id'],
                "initial_slot": result['initial_slot'],
                "rescheduled": result['rescheduled']
            })
            logger.info(f"Completed processing for CV: {cv_file}")
        except ValueError as e:
            cv_results.append({
                "cv_file": cv_file,
                "error": str(e)
            })
            logger.error(f"Error processing CV {cv_file}: {e}")



# Filter and display shortlisted candidates
shortlisted_candidates = [result for result in cv_results if result.get('shortlisted', False)]
logger.info(f"Total candidates processed: {len(cv_results)}, Shortlisted: {len(shortlisted_candidates)}")
if shortlisted_candidates:
    logger.info("Shortlisted candidates:")
    for candidate in shortlisted_candidates:
        logger.info(f"- {candidate['cv_file']}: {candidate['cv']['candidate_name']} (Score: {candidate['match_score']}, Slot: {candidate['initial_slot']})")
else:
    logger.info("No candidates were shortlisted.")

print("\nShortlisted Candidates:")
print(json.dumps(shortlisted_candidates, indent=2))
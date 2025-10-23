from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import google.generativeai as genai
from fastapi.middleware.cors import CORSMiddleware
import os

# Load env vars
load_dotenv()

# Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Initialize app
app = FastAPI(title="MedAI Progressive Chatbot")
# ✅ Allow all origins (or restrict later)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["http://127.0.0.1:5500"] for stricter
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request model
class ChatRequest(BaseModel):
    message: str
    session_id: str

# Store chat states
chat_histories = {}
user_steps = {}

# Base prompt
SYSTEM_PROMPT = (
    "You are MedAI — a medically aware, ethical conversational assistant. "
    "You only provide general medical information and safe guidance. "
    "Never diagnose or prescribe medicine. Always stay concise and calm. "
    "Ask only one medically relevant question at a time to understand the user's condition. "
    "Once you have enough information, summarize briefly and suggest professional consultation. "
    "If any symptom seems urgent (like chest pain, difficulty breathing, or bleeding), advise immediate medical attention."
)

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        session_id = request.session_id
        user_message = request.message.strip()

        # Initialize new session
        if session_id not in chat_histories:
            chat_histories[session_id] = []
            user_steps[session_id] = 1

        history = chat_histories[session_id]
        step = user_steps[session_id]

        # Store user message
        history.append({"role": "user", "text": user_message})

        # Build conversation for Gemini
        contents = [
            {"role": "user", "parts": [{"text": SYSTEM_PROMPT}]}
        ]

        for msg in history:
            role = "user" if msg["role"] == "user" else "model"
            contents.append({"role": role, "parts": [{"text": msg["text"]}]})

        # Add stage instruction
        if step == 1:
            instruction = "Ask one simple question about the symptom (like location, type, or severity)."
        elif step == 2:
            instruction = "Ask one follow-up question about when it started or how long it's been happening."
        elif step == 3:
            instruction = "Ask one question about any related symptoms."
        elif step == 4:
            instruction = "Ask one question about whether the user has tried any remedies or treatments."
        elif step == 5:
            instruction = "Ask one last question about whether this has happened before or is new."
        else:
            instruction = "Now acknowledge it and suggest the first aid treatment to him what paitent can do himself."

        # Add the instruction for the model
        contents.append({
            "role": "user",
            "parts": [{"text": f"User said: {user_message}\n\n{instruction}"}]
        })

        # Generate response
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(contents)
        reply_text = response.text.strip()

        # Save model response
        history.append({"role": "model", "text": reply_text})

        # Increment step
        user_steps[session_id] = step + 1

        return JSONResponse({
            "reply": reply_text,
            "step": step,
            "session_id": session_id
        })

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/")
def root():
    return {"message": "MedAI Progressive Chatbot API is running!"}



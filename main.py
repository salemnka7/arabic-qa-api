from fastapi import FastAPI, UploadFile, File, Form, Depends
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from pathlib import Path
import shutil
import os
import hashlib
import pickle

# Import logic from app_logic.py
from app_logic import (
    extract_text_from_pdf, extract_text_from_docx, extract_text_from_txt,
    normalize_arabic_text, create_and_save_vector_db, load_vector_db,
    get_answer_from_openai, load_users, save_users, hash_password, authenticate, add_user
)

app = FastAPI()

# Enable CORS for any frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "data/files"
Path(UPLOAD_DIR).mkdir(parents=True, exist_ok=True)

# ==== AUTHENTICATION ====
class LoginRequest(BaseModel):
    username: str
    password: str

class RegisterRequest(BaseModel):
    username: str
    password: str
    role: str  # "user" or "admin"

@app.post("/login")
def login(data: LoginRequest):
    role = authenticate(data.username, data.password)
    if role:
        return {"message": "Login successful", "role": role}
    return JSONResponse(status_code=401, content={"message": "Invalid credentials"})

@app.post("/register")
def register(data: RegisterRequest):
    success = add_user(data.username, data.password, data.role)
    if success:
        return {"message": f"User {data.username} created successfully"}
    else:
        return JSONResponse(status_code=409, content={"message": "Username already exists"})

@app.get("/users")
def get_users():
    users = load_users()
    return [{"username": u, "role": info["role"]} for u, info in users.items()]

# ==== FILE UPLOAD ====
@app.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    all_text = ""
    for file in files:
        ext = Path(file.filename).suffix.lower()
        path = f"{UPLOAD_DIR}/{file.filename}"
        with open(path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        if ext == ".pdf":
            text = extract_text_from_pdf(path)
        elif ext == ".docx":
            text = extract_text_from_docx(path)
        elif ext == ".txt":
            text = extract_text_from_txt(path)
        else:
            continue

        all_text += text + "\n"

    normalized = normalize_arabic_text(all_text)
    try:
        create_and_save_vector_db(normalized)
        return {"message": "Documents uploaded and processed successfully"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": str(e)})

# ==== ASK A QUESTION ====
class Question(BaseModel):
    query: str

@app.post("/ask")
def ask(data: Question):
    db = load_vector_db()
    if db:
        docs = db.similarity_search(data.query, k=4)
        context = "\n\n".join([doc.page_content for doc in docs])
        response = get_answer_from_openai(data.query, context, openai_client=None)
        return {"answer": response}
    return JSONResponse(status_code=404, content={"message": "No vector database found. Please upload and process documents first."})
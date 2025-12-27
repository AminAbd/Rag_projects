from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import time
import secrets

from sql_rag_agent import run_question, answer_chain, ROLE_PASSWORDS, ROLE_POLICY

# Initialize FastAPI app
app = FastAPI(
    title="SQL-RAG Agent API",
    description="Natural language to SQL agent with role-based access control",
    version="1.0.0"
)

# Enable CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# HTTP Basic Auth for password authentication
security = HTTPBasic()

# Request/Response models
class QuestionRequest(BaseModel):
    question: str

class QuestionResponse(BaseModel):
    question: str
    sql: str
    rows: Optional[str] = None
    answer: Optional[str] = None
    execution_time: float
    role: str
    error: Optional[str] = None

class AuthResponse(BaseModel):
    authenticated: bool
    role: Optional[str] = None
    message: str

# Helper function to verify password and get role
def verify_password(credentials: HTTPBasicCredentials = Depends(security)):
    """Verify password and return role."""
    password = credentials.password
    
    # Check which role this password belongs to
    for role, role_password in ROLE_PASSWORDS.items():
        if password == role_password:
            return role
    
    # Password didn't match any role
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Incorrect password",
        headers={"WWW-Authenticate": "Basic"},
    )

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "SQL-RAG Agent API",
        "version": "1.0.0",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "auth": "/auth",
            "query": "/query"
        }
    }

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "SQL-RAG Agent"}

# Authentication endpoint
@app.post("/auth", response_model=AuthResponse)
async def authenticate(credentials: HTTPBasicCredentials = Depends(security)):
    """Authenticate user and return role."""
    password = credentials.password
    
    # Check which role this password belongs to
    for role, role_password in ROLE_PASSWORDS.items():
        if password == role_password:
            role_display = role.replace('_', ' ').title()
            return AuthResponse(
                authenticated=True,
                role=role,
                message=f"Authentication successful! Role: {role_display}"
            )
    
    # Password didn't match
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Incorrect password",
        headers={"WWW-Authenticate": "Basic"},
    )

# Query endpoint
@app.post("/query", response_model=QuestionResponse)
async def query_database(
    request: QuestionRequest,
    role: str = Depends(verify_password)
):
    """
    Execute a natural language query against the database.
    
    Requires HTTP Basic Authentication with role-based password.
    - Admin password: admin123 (full access)
    - Support Agent password: support123 (restricted access)
    """
    if not request.question or not request.question.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question cannot be empty"
        )
    
    # Record start time
    start_time = time.time()
    
    # Run the question
    sql, rows, err = run_question(request.question.strip(), role)
    
    # Calculate execution time
    execution_time = time.time() - start_time
    
    # If there's an error, return error response
    if err:
        return QuestionResponse(
            question=request.question,
            sql=sql if sql else "",
            error=err,
            execution_time=execution_time,
            role=role
        )
    
    # Generate natural language answer
    try:
        answer = answer_chain.invoke({
            "question": request.question,
            "sql": sql,
            "rows": rows
        })
    except Exception as e:
        answer = f"Error generating answer: {str(e)}"
    
    return QuestionResponse(
        question=request.question,
        sql=sql,
        rows=str(rows) if rows else None,
        answer=answer,
        execution_time=execution_time,
        role=role
    )

# Get available roles endpoint
@app.get("/roles")
async def get_roles():
    """Get information about available roles and their permissions."""
    roles_info = {}
    for role, policy in ROLE_POLICY.items():
        roles_info[role] = {
            "allowed_tables": list(policy["allowed_tables"]),
            "forbidden_patterns": policy["forbidden"],
            "has_restrictions": len(policy["forbidden"]) > 0
        }
    return {
        "roles": roles_info,
        "note": "Passwords are required for authentication. Use /auth endpoint to authenticate."
    }

# Example questions endpoint
@app.get("/examples")
async def get_example_questions():
    """Get example questions users can ask."""
    return {
        "examples": [
            "Which 5 customers spent the most money?",
            "What are the top 10 best-selling tracks?",
            "List all albums by AC/DC",
            "How many invoices were created in 2010?",
            "What is the total revenue by country?",
            "Which artists have the most albums?",
            "What are the most popular genres?",
            "List all playlists and their track counts"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)


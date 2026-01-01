import os
import uvicorn
from pydantic import BaseModel
from database.db import Database
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from services.auth_service import AuthService
from fastapi.middleware.cors import CORSMiddleware
from services.face_service import FaceRecognitionService
from services.enhanced_liveness_service import EnhancedLivenessService

app = FastAPI(
    title="Face Authentication System",
    description="CNN-based face authentication with liveness detection",
    version="2.0.13"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# initialize services
face_service = FaceRecognitionService()
liveness_service = EnhancedLivenessService(face_service)
auth_service = AuthService()
db = Database()

# pydantic
class RegisterRequest(BaseModel):
    username: str
    email: Optional[str] = None
    image: str

class AuthenticateRequest(BaseModel):
    image: str

class LivenessRequest(BaseModel):
    frames: List[str]

class TokenRequest(BaseModel):
    token: str

# route
@app.get("/", response_class=HTMLResponse)
async def read_root():
    """
    Serve the main HTML page
    """
    with open("templates/index.html", "r", encoding="utf-8") as f:
        return f.read()
    
@app.get("api/health")
async def health_check():
    """
    health check endpoint
    """
    return {
        "status": "healthy",
        "service": "face-authentication",
        "device": str(face_service.device)
    }

@app.post("/api/register")
async def register(request: RegisterRequest):
    try:
        # dat ten phai tu 3 - 50 ki tu nha <3
        if len(request.username) < 3 or len(request.username) > 50:
            raise HTTPException(400, "Username must be 3-50 characters")
        
        # ten trung = getout
        if db.get_user_by_username(request.username):
            raise HTTPException(409, "Username already exists")
        
        # extract face embedding
        print(f"Extracting face embedding for {request.username}...")
        face_embedding = face_service.extract_face_embedding(request.image)

        if face_embedding is None:
            raise HTTPException(400, "No face detected in image")
        
        # check if face already registered
        all_users = db.get_all_users()
        for user in all_users:
            similarity = face_service.calculate_similarity(
                face_embedding,
                user['face_encoding']
            )
            if similarity >= 0.6:
                raise HTTPException(409, f"Face already registered to user: {user['username']}")

        # save user
        user_id = db.create_user(
            username=request.username,
            email=request.email,
            face_encoding=face_embedding.tolist()
        )

        # generate token
        token = auth_service.generate_token(user_id, request.username)

        print(f"User registered: {request.username}")

        return {
            "success": True,
            "message": "User registered successfully",
            "user_id": user_id,
            "username": request.username,
            "token": token
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"Registration error: {str(e)}")
        raise HTTPException(500, f"Registration failed: {str(e)}")

@app.post("/api/authenticate")
async def authenticate(request: AuthenticateRequest):
    """
    authenticate user with face
    """
    try:
        print("Extracting face embedding for authentication...")
        face_embedding = face_service.extract_face_embedding(request.image)

        if face_embedding is None:
            raise HTTPException(400, "No face detected in image")
        
        # find matching user
        all_users = db.get_all_users()
        best_match = None
        best_similarity = -1

        for user in all_users:
            similarity = face_service.calculate_similarity(
                face_embedding,
                user['face_encoding']
            )

            if similarity > best_similarity:
                best_similarity = similarity
                best_match = user

        # check threshold
        if best_similarity < 0.6:
            raise HTTPException(401, "Face not recognized")
        
        # update last login
        db.update_last_login(best_match['user_id'])

        # generate token
        token = auth_service.generate_token(
            best_match['user_id'],
            best_match['username']
        )

        print(f"User authenticated: {best_match['username']} (similarity: {best_similarity:.2f})")

        return {
            "success": True,
            "message": "Authentication successful",
            "user_id": best_match['user_id'],
            "username": best_match['username'],
            "similarity": best_similarity,
            "token": token
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"Authentication error: {str(e)}")
        raise HTTPException(500, f"Authentication failed: {str(e)}")

@app.post("/api/liveness-check")
async def liveness_check(request: LivenessRequest):
    """ 
    check liveness from multiple frames
    """
    try:
        if len(request.frames) < 3:
            raise HTTPException(400, "Minimum 3 frames required")
        
        print(f"Performing liveness check with {len(request.frames)} frames...")
        result = liveness_service.check_liveness(request.frames)

        print(f"Liveness check result: {result['is_live']}")

        return result
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"Liveness check error: {str(e)}")
        raise HTTPException(500, f"Liveness check failed: {str(e)}")
    
@app.post("/api/verify-token")
async def verify_token(request: TokenRequest):
    """Verify JWT token"""
    try:
        payload = auth_service.verify_token(request.token)
        
        if payload is None:
            raise HTTPException(401, "Invalid or expired token")
        
        return {
            "valid": True,
            "user_id": payload['user_id'],
            "username": payload['username']
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))
    
@app.get("/api/users")
async def get_users():
    """Get all users"""
    try:
        users = db.get_all_users()
        # Remove face encodings from response
        for user in users:
            user.pop('face_encoding', None)
        
        return {
            "total_users": len(users),
            "users": users
        }
    except Exception as e:
        raise HTTPException(500, str(e))


if __name__ == "__main__":
    print("🚀 Starting Face Authentication System...")
    print(f"📊 Device: {face_service.device}")
    print("🌐 Server: http://localhost:8000")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
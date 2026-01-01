import uvicorn
from main import app

if __name__ == "__main__":
    print("Starting face authentication web application...")
    print("🌐 http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
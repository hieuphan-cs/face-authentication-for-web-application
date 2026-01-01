# step 1: open folder
cd face_authentication_for_web_application

# step 2: open file requirements.txt, then download all necessary modules
pip install fastapi uvicorn torch torchvision facenet-pytorch Pillow numpy PyJWT python-multipart
Notice: open dlib-19.24.99-cp312-cp312-win_amd64.whl to download 

# step 3: open run.py then run this file
python -m uvicorn run:app --reload --port 8000

# step 4: open link: http://localhost:8000

## Author
hieuphan-cs

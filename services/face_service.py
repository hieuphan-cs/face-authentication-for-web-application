import io
import torch
import base64
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1

class FaceRecognitionService:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Initializing face recognition on {self.device}")

        self.mtcnn = MTCNN(
            keep_all=False,
            device=self.device,
            min_face_size=40,
            thresholds=[0.6, 0.7, 0.7],
            post_process=True
        )

        self.resnet = InceptionResnetV1(
            pretrained='vggface2',
        ).eval().to(self.device)
        
    def decode_base64_image(self, base64_string):
        """
        Decode base64 image to PIL Image 
        """
        try:
            if 'base64,' in base64_string:
                base64_string = base64_string.split('base64,')[1]

            image_data = base64.b64decode(base64_string)
            image = Image.open(io.BytesIO(image_data))

            return image.convert('RGB')
        except Exception as e:
            raise ValueError(f"Invalid image data: {str(e)}")
        
    def extract_face_embedding(self, image_data):
        """
        Extract face embedding using MTCNn and FaceNet

        :param image_data: base64 encoded image string or PIL Image
        :returns: 512-dim face embedding vector or none if no face is detected
        """
        try:
            if isinstance(image_data, str):
                image = self.decode_base64_image(image_data)
            else:
                image = image_data
            
            # detect and align face using MTCNN
            face_tensor = self.mtcnn(image)

            if face_tensor is None:
                print("No face detected")
                return None
            
            # get face embedding using FaceNet
            face_tensor = face_tensor.unsqueeze(0).to(self.device)

            with torch.no_grad():
                embedding = self.resnet(face_tensor)

            embedding_np = embedding.cpu().numpy().flatten()
            
            print(f"Face embedding extracted: {embedding_np.shape}")
            return embedding_np

        except Exception as e:
            print(f"Error extracting face embedding: {str(e)}")
            return None

    def calculate_similarity(self, embedding1, embedding2):
        """
        Calculate cosine similarity between two embeddings
        
        :param embedding1: first embedding
        :param embedding2: second embedding
        """
        if isinstance(embedding1, list):
            embedding1 = np.array(embedding1)
        if isinstance(embedding2, list):
            embedding2 = np.array(embedding2)

        # normalize embedding
        emb1_norm = embedding1 / np.linalg.norm(embedding1)
        emb2_norm = embedding2 / np.linalg.norm(embedding2)

        # calculate cosine similarity
        similarity = np.dot(emb1_norm, emb2_norm)

        return float(similarity)


# # example
# with open("imgs/test_picture1.jpg", "rb") as f:
#     img_bytes = f.read()
# img_b64 = base64.b64encode(img_bytes).decode()
# face_ser = FaceRecognitionService()
# embedding1 = face_ser.extract_face_embedding(image_data=img_b64)

# with open("imgs/pic3.jpg", "rb") as f:
#     img_bytes = f.read()
# img_b64 = base64.b64encode(img_bytes).decode()
# face_ser = FaceRecognitionService()
# embedding2 = face_ser.extract_face_embedding(image_data=img_b64)

# face_service = FaceRecognitionService()
# a = face_service.calculate_similarity(embedding1, embedding2)
# print(a)
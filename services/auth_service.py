import jwt
from datetime import datetime, timedelta, timezone, UTC

class AuthService:
    def __init__(self):
        self.secret_key = ""
        self.algorithm = "HS256"
        self.expiration_hours = 24

    def generate_token(self, user_id, username):
        """
        generate jwt token
        """
        payload = {
            'user_id': user_id,
            'username': username,
            'exp': datetime.now(timezone.utc) + timedelta(hours=self.expiration_hours),
            'iat': datetime.now(timezone.utc)
        }

        token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
        
        return token
    
    def verify_token(self, token):
        """
        verify token (JWT)
        """
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None
import os
import json
import uuid
from datetime import datetime, timezone

class Database:
    def __init__(self, db_file="database/users.json"):
        self.db_file = db_file
        self._ensure_db_exists()
    
    def _ensure_db_exists(self):
        """
        create database file if it doesn't exist
        """
        os.makedirs(os.path.dirname(self.db_file), exist_ok=True)
        if not os.path.exists(self.db_file):
            with open(self.db_file, 'w') as f:
                json.dump({"users":[]}, f)

    def _read_db(self):
        with open(self.db_file, 'r') as f:
            return json.load(f)

    def _write_db(self, data):
        with open(self.db_file, 'w') as f:
            json.dump(data, f, indent=2)

    def create_user(self, username, email, face_encoding):
        db = self._read_db()

        user = {
            "user_id": str(uuid.uuid4()),
            "username": username,
            "email": email,
            "face_encoding": face_encoding,
            "registered_at": datetime.now(timezone.utc).isoformat(),
            "last_login": None
        }

        db["users"].append(user)
        self._write_db(db)

        return user["user_id"]
    
    def get_user_by_username(self, username):
        db = self._read_db()
        for user in db["users"]:
            if user["username"] == username:
                return user
        return None
    
    def get_user_by_id(self, user_id):
        db = self._read_db()
        for user in db["users"]:
            if user["user_id"] == user_id:
                return user
        return None

    def update_last_login(self, user_id):
        db = self._read_db()
        for user in db["users"]:
            if user["user_id"] ==  user_id:
                user["last_login"] = datetime.now(timezone.utc).isoformat()
                break
        self._write_db(db)

    def get_all_users(self):
        db = self._read_db()
        return db["users"]
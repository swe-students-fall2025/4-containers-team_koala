import os
from datetime import datetime
from pymongo import MongoClient
from bson import ObjectId
from dotenv import load_dotenv


load_dotenv()

MONGO_URI = os.getenv("MONGO_URI", "mongodb://mongo:27017/ASL_DB")
DB_NAME = os.getenv("MONGO_DB_NAME", "ASL_DB")

client = MongoClient(MONGO_URI)
db = client[DB_NAME]
users = db.users
assessments = db.assessments

user_doc = {
    "_id": ObjectId(),
    "username": "temp",
    "email": "temp@gmail.com",
    "password_hash": "TempPass",
    "created_at": datetime.utcnow(),
    "last_login": None,
    "progress": {"lessons_completed": [], "assessments_taken": []},
}

assessments_doc = {
    "_id": ObjectId(),
    "title": "Sample Assessment",
    "num_questions": 10,
    "questions": [],
}

result = users.insert_one(user_doc)
print("Inserted user with _id:", result.inserted_id)

result1 = assessments.insert_one(assessments_doc)
print("Inserted assessment with _id:", result1.inserted_id)

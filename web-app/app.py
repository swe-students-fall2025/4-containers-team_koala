from flask import Flask
from pymongo import MongoClient
from dotenv import load_dotenv
import os

def create_app():
    load_dotenv()

    app = Flask(__name__)
    app.config["SECRET_KEY"] = os.getenv("FLASK_SECRET_KEY", "dev-secret")  # replace later

    # Setup MongoDB connection
    MONGO_URI = os.getenv("MONGO_URI", "mongodb://mongo:27017/ASL_DB")
    MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "ASL_DB")
    app.db = MongoClient(MONGO_URI).get_database(MONGO_DB_NAME)

    # Register blueprints
    from routes.auth import auth_bp
    from routes.training import training_bp
    from routes.dashboard import dashboard_bp
    from routes.training import training

    app.register_blueprint(auth_bp)
    app.register_blueprint(training_bp)
    app.register_blueprint(dashboard_bp)
    app.register_blueprint(training)


    return app

app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

from flask import Flask, redirect, url_for, session, render_template
from pymongo import MongoClient
from dotenv import load_dotenv
import os

def create_app():
    load_dotenv()
    """Application factory for ASL Practice app."""
    app = Flask(__name__)
    app.config["SECRET_KEY"] = os.getenv("FLASK_SECRET_KEY", "dev-secret")  # replace later

    # Setup MongoDB connection
    MONGO_URI = os.getenv("MONGO_URI", "mongodb://mongo:27017/ASL_DB")
    MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "ASL_DB")
    app.db = MongoClient(MONGO_URI).get_database(MONGO_DB_NAME)

    # Import and register blueprints
    from routes.auth import auth as auth_bp
    from routes.dashboard import dashboard as dashboard_bp
    from routes.training import training as training_bp

    app.register_blueprint(auth_bp)
    app.register_blueprint(dashboard_bp)
    app.register_blueprint(training_bp)
    @app.route('/')
    def home():
        return render_template('home.html')

    return app



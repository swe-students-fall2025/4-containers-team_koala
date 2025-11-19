from flask import Flask, redirect, url_for, session, render_template
from pymongo import MongoClient
from dotenv import load_dotenv
import os


def create_app():
    load_dotenv()
    """Application factory for ASL Practice app."""
    app = Flask(__name__)
    app.config["SECRET_KEY"] = os.getenv("FLASK_SECRET_KEY", "dev-secret")

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

    # ----------------------
    # Home Route Fix
    # ----------------------
    @app.route("/")
    def home():
        # If user is logged in, go to dashboard
        if "user_id" in session:
            return redirect(
                url_for("dashboard.home")
            )  # your dashboard view is named 'home'
        # Otherwise show landing page
        return render_template("home.html")

    return app

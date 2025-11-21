"""
web-app package initializer.
Sets up Flask app, database, and routes.
"""

import os

from flask import Flask, redirect, url_for, session, render_template
from pymongo import MongoClient
from dotenv import load_dotenv


def create_app():
    """Create and configure the Flask application."""
    load_dotenv()
    app = Flask(__name__)
    app.config["SECRET_KEY"] = os.getenv("FLASK_SECRET_KEY", "dev-secret")

    # Setup MongoDB connection
    mongo_uri = os.getenv("MONGO_URI", "mongodb://mongo:27017/ASL_DB")
    mongo_db_name = os.getenv("MONGO_DB_NAME", "ASL_DB")
    app.db = MongoClient(mongo_uri).get_database(mongo_db_name)

    # Import and register blueprints
    from .routes.auth import auth as auth_bp  # pylint: disable=import-outside-toplevel
    from .routes.dashboard import (
        dashboard as dashboard_bp,
    )  # pylint: disable=import-outside-toplevel
    from .routes.training import (
        training as training_bp,
    )  # pylint: disable=import-outside-toplevel

    app.register_blueprint(auth_bp)
    app.register_blueprint(dashboard_bp)
    app.register_blueprint(training_bp)

    # ----------------------
    # Home Route Fix
    # ----------------------
    @app.route("/")
    def home():
        """Landing page – redirect to dashboard if logged in."""
        # If user is logged in, go to dashboard
        if "user_id" in session:
            return redirect(
                url_for("dashboard.home")
            )  # your dashboard view is named 'home'
        # Otherwise show landing page
        return render_template("home.html")

    return app

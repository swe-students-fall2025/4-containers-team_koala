from flask import Flask
from pymongo import MongoClient
import os

def create_app():
    app = Flask(__name__)
    app.config["SECRET_KEY"] = "dev_key"  # replace later

    # MongoDB will be provided by teammate
    MONGO_URI = os.getenv("MONGO_URI", "mongodb://mongodb:27017/asl")
    app.db = MongoClient(MONGO_URI).get_database()

    # Register blueprints
    from routes.auth import auth_bp
    from routes.training import training_bp
    from routes.dashboard import dashboard_bp

    app.register_blueprint(auth_bp)
    app.register_blueprint(training_bp)
    app.register_blueprint(dashboard_bp)

    return app

app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

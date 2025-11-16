from flask import Flask, redirect, url_for, session, render_template

def create_app():
    """Application factory for ASL Practice app."""
    app = Flask(__name__)
    app.secret_key = "your_secret_key"  # replace with environment variable in production

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



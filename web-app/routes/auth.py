"""Authentication routes for the ASL practice web app.

This module handles user login, logout, and registration.
"""

from flask import Blueprint, request, redirect, url_for, flash, session, render_template, current_app
from werkzeug.security import generate_password_hash, check_password_hash
from bson import ObjectId
from datetime import datetime

auth = Blueprint('auth', __name__)


def validate_login(username: str, password: str) -> bool:
    """Validate a user's login credentials.

    Args:
        username (str): The username provided.
        password (str): The password provided.

    Returns:
        bool: True if credentials are valid, False otherwise.
    """
    db = current_app.db
    users = db.users

    user = users.find_one({"username": username})
    if not user:
        return False
    if not check_password_hash(user["password_hash"], password):
        return False
    return True

def validate_registration(email:str, username: str, password: str) -> tuple[bool, str]:
    """Validate registration input.

    Args:
        email (str): The email to register.
        username (str): The username to register.
        password (str): The password to register.

    Returns:
        tuple[bool, str]: (True, "") if valid; otherwise (False, reason)
    """

    db = current_app.db
    users = db.users
    
    email = (email or "").strip().lower()
    username = (username or "").strip()
    if not username or not password or not email:
        return False, "Username, email, and password cannot be empty."
    if len(password) < 6:
        return False, "Password must be at least 6 characters long."
    if users.find_one({"username": username}):
        return False, "Username already exists."
    if users.find_one({"email": email}):
        return False, "Email already registered."
    return True, ""


@auth.route('/login', methods=['GET', 'POST'])
def login():
    """Handle user login."""
    if request.method == 'POST':
        username = (request.form.get('username') or "").strip()
        password = request.form.get('password')
 
        if validate_login(username, password):
            session['user_id'] = username
            db = current_app.db
            users = db.users
            users.update_one({"username": username}, {"$set": {"last_login": datetime.utcnow()}})
            flash("Logged in successfully.", "success")
            return redirect(url_for('dashboard.home'))
        else:
            flash("Invalid username or password.", "danger")
            return render_template('login.html')

        # ✔ Store ObjectId as string in session
        session['user_id'] = str(user['_id'])

        # Update last login
        db['users'].update_one(
            {"_id": user["_id"]},
            {"$set": {"last_login": datetime.utcnow()}}
        )

        flash("Logged in successfully.", "success")
        return redirect(url_for('dashboard.home'))

    return render_template('login.html')


@auth.route('/logout')
def logout():
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for('auth.login'))


@auth.route('/register', methods=['GET', 'POST'])
def register():
    """Handle new user registration."""
    if request.method == 'POST':
        username = (request.form.get('username') or "").strip()
        email = (request.form.get('email') or "").strip().lower()
        password = request.form.get('password')

        valid, message = validate_registration(email, username, password)
        if not valid:
            flash(message, "danger")
            return render_template('register.html')

        db = current_app.db
        users = db.users
        users.insert_one({
            "username": username,
            "email": email,
            "password_hash": generate_password_hash(password),
            "created_at": datetime.utcnow(),
            "last_login": None,
            "progress": {
                "lessons_completed": [],
                "assessments_taken": []
            }
        })
        flash("Registration successful! You can now log in.", "success")
        return redirect(url_for('auth.login'))

    return render_template('register.html')

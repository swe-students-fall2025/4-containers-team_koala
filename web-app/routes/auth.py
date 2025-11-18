from flask import Blueprint, request, redirect, url_for, flash, session, render_template, current_app
from werkzeug.security import generate_password_hash, check_password_hash
from bson import ObjectId
from datetime import datetime

auth = Blueprint('auth', __name__)


@auth.route('/login', methods=['GET', 'POST'])
def login():
    """Handle user login."""
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        db = current_app.db
        user = db['users'].find_one({"username": username})

        if not user or not check_password_hash(user['password_hash'], password):
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
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')

        db = current_app.db

        # Check if username already exists
        if db['users'].find_one({"username": username}):
            flash("Username already exists.", "danger")
            return render_template('register.html')

        # Create user document
        user_doc = {
            "username": username,
            "email": email,
            "password_hash": generate_password_hash(password),
            "created_at": datetime.utcnow(),
            "last_login": None,
            "progress": {
                "lessons_completed": [],
                "assessments_taken": []
            }
        }

        result = db['users'].insert_one(user_doc)

        flash("Registration successful! Please log in.", "success")
        return redirect(url_for('auth.login'))

    return render_template('register.html')

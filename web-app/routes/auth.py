"""Authentication routes for the ASL practice web app.

This module handles user login, logout, and registration.
"""

from flask import Blueprint, request, redirect, url_for, flash, session, render_template
from werkzeug.security import generate_password_hash, check_password_hash

auth = Blueprint('auth', __name__)

# Example in-memory user store (to be replaced by database)
users = {}


def validate_login(username: str, password: str) -> bool:
    """Validate a user's login credentials.

    Args:
        username (str): The username provided.
        password (str): The password provided.

    Returns:
        bool: True if credentials are valid, False otherwise.
    """
    user = users.get(username)
    if not user:
        return False
    return check_password_hash(user['password_hash'], password)


def validate_registration(username: str, password: str) -> tuple[bool, str]:
    """Validate registration input.

    Args:
        username (str): The username to register.
        password (str): The password to register.

    Returns:
        tuple[bool, str]: (True, "") if valid; otherwise (False, reason)
    """
    if username in users:
        return False, "Username already exists."
    if not username or not password:
        return False, "Username and password cannot be empty."
    if len(password) < 6:
        return False, "Password must be at least 6 characters long."
    return True, ""


@auth.route('/login', methods=['GET', 'POST'])
def login():
    """Handle user login.

    GET: Display the login form.
    POST: Validate credentials and log the user in.
    """
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        if validate_login(username, password):
            session['user_id'] = username
            flash("Logged in successfully.", "success")
            return redirect(url_for('dashboard.home'))
        else:
            flash("Invalid username or password.", "danger")

    return render_template('login.html')


@auth.route('/logout')
def logout():
    """Log the current user out and clear the session."""
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for('auth.login'))


@auth.route('/register', methods=['GET', 'POST'])
def register():
    """Handle new user registration.

    GET: Display the registration form.
    POST: Create a new user in the database (to be implemented).
    """
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        valid, message = validate_registration(username, password)
        if not valid:
            flash(message, "danger")
            return render_template('register.html')

        # Store password hash in memory for now (replace with DB later)
        users[username] = {'password_hash': generate_password_hash(password)}
        flash("Registration successful! You can now log in.", "success")
        return redirect(url_for('auth.login'))

    return render_template('register.html')

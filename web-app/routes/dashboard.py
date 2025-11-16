from flask import Blueprint, render_template, session, redirect, url_for

dashboard = Blueprint('dashboard', __name__, url_prefix='/dashboard')

@dashboard.route('/')
def home():
    if 'user_id' not in session:
        return redirect(url_for('auth.login'))

    # placeholder progress data
    progress = {
        "lessons_completed": ["lesson1", "lesson2"],
        "assessments_passed": ["assessment1"]
    }
    return render_template('dashboard.html', progress=progress)

from flask import Blueprint, redirect, render_template, session, url_for

training = Blueprint('training', __name__, url_prefix='/training')

@training.route('/')
def lessons():
    if 'user_id' not in session:
        return redirect(url_for('auth.login'))

    lessons = ["lesson1", "lesson2", "lesson3"]
    return render_template('training.html', lessons=lessons)

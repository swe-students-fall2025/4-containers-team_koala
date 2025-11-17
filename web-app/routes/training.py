from flask import Blueprint, redirect, render_template, session, url_for

training = Blueprint('training', __name__, url_prefix='/training')

# Lessons overview
@training.route('/')
def lessons():
    if 'user_id' not in session:
        return redirect(url_for('auth.login'))

    lessons = [
        {"id": 1, "title": "Lesson 1: ASL Alphabet A-G"},
        {"id": 2, "title": "Lesson 2: ASL Alphabet H-N"},
        {"id": 3, "title": "Lesson 3: ASL Alphabet O-U"},
        {"id": 4, "title": "Lesson 4: ASL Alphabet V-Z"},
        {"id": 5, "title": "Final Practice Lesson"},
    ]
    return render_template('lessons.html', lessons=lessons)

# Individual lesson page
@training.route('/lesson/<int:num>')
def lesson(num):
    if 'user_id' not in session:
        return redirect(url_for('auth.login'))

    # Simulated data per lesson
    lesson_data = {
        1: "You will learn letters A-G in ASL!",
        2: "You will learn letters H-N in ASL!",
        3: "You will learn letters O-U in ASL!",
        4: "You will learn letters V-Z in ASL!",
        5: "Test your knowledge of the entire ASL alphabet!",
    }

    if num not in lesson_data:
        return redirect(url_for('training.lessons'))  # go back if invalid lesson

    return render_template('lesson.html', lesson_num=num, description=lesson_data[num])

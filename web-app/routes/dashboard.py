from flask import Blueprint, render_template, session, redirect, url_for, current_app

dashboard = Blueprint("dashboard", __name__, url_prefix="/dashboard")


@dashboard.route("/")
def home():
    if "user_id" not in session:
        return redirect(url_for("auth.login"))

    user_id = session["user_id"]
    db = current_app.db
    assessments = list(db["assessments"].find({"user_id": user_id}))

    lessons_completed = sorted({a["lesson_id"] for a in assessments})
    assessments_passed = [a for a in assessments if a.get("passed")]

    progress = {
        "lessons_completed": lessons_completed,
        "assessments_passed": [a["lesson_id"] for a in assessments_passed],
    }

    return render_template("dashboard.html", progress=progress)

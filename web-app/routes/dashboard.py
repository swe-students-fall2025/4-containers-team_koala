from bson import ObjectId
from flask import Blueprint, render_template, session, redirect, url_for, current_app

dashboard = Blueprint("dashboard", __name__, url_prefix="/dashboard")

@dashboard.route("/")
def home():
    if "user_id" not in session:
        return redirect(url_for("auth.login"))

    db = current_app.db
    user_id = session["user_id"]

    # Load user
    user = db["users"].find_one({"_id": ObjectId(user_id)})
    if not user:
        session.clear()
        return redirect(url_for("auth.login"))

    username = user.get("username", "User")

    # Get progress
    progress = user.get("progress", {})
    lessons_completed_ids = progress.get("lessons_completed", [])
    assessments_taken_titles = progress.get("assessments_taken", [])

    # Map lesson numbers to lesson titles
    from routes.training import LESSON_MAP
    lessons_completed = [LESSON_MAP.get(l, {}).get("title", f"Lesson {l}") for l in lessons_completed_ids]

    return render_template(
        "dashboard.html",
        username=username,
        lessons_completed=lessons_completed,
        assessments_taken=assessments_taken_titles
    )

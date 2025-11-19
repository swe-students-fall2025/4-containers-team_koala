"""Dashboard routes for user progress and overview."""

from bson import ObjectId
from flask import Blueprint, render_template, session, redirect, url_for, current_app
from routes.training import LESSON_MAP  # moved import to module level

dashboard = Blueprint("dashboard", __name__, url_prefix="/dashboard")


@dashboard.route("/")
def home():
    """Render the dashboard home page for the logged-in user."""
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
    completed_ids = progress.get("lessons_completed", [])
    assessments_taken_titles = progress.get("assessments_taken", [])

    # Map lesson numbers to titles
    lessons_completed = [
        LESSON_MAP.get(lesson_id, {}).get("title", f"Lesson {lesson_id}")
        for lesson_id in completed_ids
    ]

    return render_template(
        "dashboard.html",
        username=username,
        lessons_completed=lessons_completed,
        assessments_taken=assessments_taken_titles,
    )

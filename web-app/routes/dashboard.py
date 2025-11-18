from flask import Blueprint, render_template, session, redirect, url_for, current_app
from bson import ObjectId

dashboard = Blueprint("dashboard", __name__, url_prefix="/dashboard")


@dashboard.route("/")
def home():
    if "user_id" not in session:
        return redirect(url_for("auth.login"))

    db = current_app.db
    user_id = session["user_id"]

    # Load user for display info
    try:
        user = db["users"].find_one({"_id": ObjectId(user_id)})
    except Exception:
        # If session somehow stores a bad ID, log out safely
        session.clear()
        return redirect(url_for("auth.login"))

    if not user:
        session.clear()
        return redirect(url_for("auth.login"))

    username = user.get("username", "User")

    # Query assessments using ObjectId
    assessments = list(db["assessments"].find({
        "user_id": ObjectId(user_id)
    }))

    lessons_completed = sorted({
        a.get("lesson_id")
        for a in assessments
        if a.get("lesson_id") is not None
    })

    assessments_passed = [
        a.get("lesson_id")
        for a in assessments
        if a.get("passed") is True and a.get("lesson_id") is not None
    ]

    progress = {
        "lessons_completed": lessons_completed,
        "assessments_passed": assessments_passed,
    }

    return render_template(
        "dashboard.html",
        username=username,
        progress=progress
    )

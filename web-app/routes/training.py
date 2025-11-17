from __future__ import annotations

import time
from typing import List, Dict, Any

from flask import (
    Blueprint,
    redirect,
    render_template,
    session,
    url_for,
    current_app,
)

training = Blueprint("training", __name__, url_prefix="/training")

LESSONS: List[Dict[str, Any]] = [
    {
        "id": 1,
        "title": "Lesson 1: ASL Alphabet A–G",
        "description": "Learn and practice ASL handshapes for the letters A through G.",
        "signs": ["A", "B", "C", "D", "E", "F", "G"],
    },
    {
        "id": 2,
        "title": "Lesson 2: ASL Alphabet H–N",
        "description": "Learn and practice ASL handshapes for the letters H through N.",
        "signs": ["H", "I", "J", "K", "L", "M", "N"],
    },
    {
        "id": 3,
        "title": "Lesson 3: ASL Alphabet O–U",
        "description": "Learn and practice ASL handshapes for the letters O through U.",
        "signs": ["O", "P", "Q", "R", "S", "T", "U"],
    },
    {
        "id": 4,
        "title": "Lesson 4: ASL Alphabet V–Z",
        "description": "Learn and practice ASL handshapes for the letters V through Z.",
        "signs": ["V", "W", "X", "Y", "Z"],
    },
    {
        "id": 5,
        "title": "Final Practice Lesson",
        "description": "Review all letters A–Z and test your recognition skills.",
        "signs": list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
    },
]

LESSON_MAP = {lesson["id"]: lesson for lesson in LESSONS}

# For each lesson, define an assessment:
# - tasks: prompts + target signs
# - time_window_seconds: how far back we look in detections
ASSESSMENTS: Dict[int, Dict[str, Any]] = {
    1: {
        "time_window_seconds": 60,
        "tasks": [
            {
                "prompt": "Sign the letter A three times.",
                "target_sign": "A",
                "min_repetitions": 3,
                "min_confidence": 0.8,
            },
            {
                "prompt": "Sign the letter C three times.",
                "target_sign": "C",
                "min_repetitions": 3,
                "min_confidence": 0.8,
            },
        ],
    },
    2: {
        "time_window_seconds": 60,
        "tasks": [
            {
                "prompt": "Sign the letter H three times.",
                "target_sign": "H",
                "min_repetitions": 3,
                "min_confidence": 0.8,
            },
            {
                "prompt": "Sign the letter N three times.",
                "target_sign": "N",
                "min_repetitions": 3,
                "min_confidence": 0.8,
            },
        ],
    },
    3: {
        "time_window_seconds": 60,
        "tasks": [
            {
                "prompt": "Sign the letter O three times.",
                "target_sign": "O",
                "min_repetitions": 3,
                "min_confidence": 0.8,
            },
            {
                "prompt": "Sign the letter S three times.",
                "target_sign": "S",
                "min_repetitions": 3,
                "min_confidence": 0.8,
            },
        ],
    },
    4: {
        "time_window_seconds": 60,
        "tasks": [
            {
                "prompt": "Sign the letter V three times.",
                "target_sign": "V",
                "min_repetitions": 3,
                "min_confidence": 0.8,
            },
            {
                "prompt": "Sign the letter Z three times.",
                "target_sign": "Z",
                "min_repetitions": 3,
                "min_confidence": 0.8,
            },
        ],
    },
    5: {
        "time_window_seconds": 90,
        "tasks": [
            {
                "prompt": "Sign the letter A three times.",
                "target_sign": "A",
                "min_repetitions": 3,
                "min_confidence": 0.8,
            },
            {
                "prompt": "Sign the letter M three times.",
                "target_sign": "M",
                "min_repetitions": 3,
                "min_confidence": 0.8,
            },
            {
                "prompt": "Sign the letter Z three times.",
                "target_sign": "Z",
                "min_repetitions": 3,
                "min_confidence": 0.8,
            },
        ],
    },
}


# --------- ROUTES ----------


@training.route("/")
def lessons() -> str:
    """Show list of lessons."""
    if "user_id" not in session:
        return redirect(url_for("auth.login"))

    return render_template("lessons.html", lessons=LESSONS)


@training.route("/lesson/<int:num>")
def lesson(num: int) -> str:
    """Show a single lesson page."""
    if "user_id" not in session:
        return redirect(url_for("auth.login"))

    lesson_obj = LESSON_MAP.get(num)
    if lesson_obj is None:
        return redirect(url_for("training.lessons"))

    return render_template(
        "lesson.html",
        lesson_num=num,
        description=lesson_obj["description"],
        signs=lesson_obj["signs"],
    )


@training.route("/lesson/<int:num>/assessment")
def assessment(num: int) -> str:
    """Run an assessment for a given lesson based on recent ML detections."""
    if "user_id" not in session:
        return redirect(url_for("auth.login"))

    user_id = session["user_id"]
    lesson_obj = LESSON_MAP.get(num)
    assessment_def = ASSESSMENTS.get(num)

    if lesson_obj is None or assessment_def is None:
        return redirect(url_for("training.lessons"))

    db = current_app.db
    detections_coll = db["detections"]

    now = time.time()
    window_start = now - assessment_def["time_window_seconds"]

    # Get recent detections for this user
    recent_detections = list(
        detections_coll.find(
            {
                "user_id": user_id,
                "timestamp": {"$gte": window_start},
            }
        )
    )

    task_results = []
    for task in assessment_def["tasks"]:
        target = task["target_sign"]
        min_rep = task["min_repetitions"]
        min_conf = task["min_confidence"]

        count = sum(
            1
            for d in recent_detections
            if d.get("sign_label") == target
            and float(d.get("confidence", 0.0)) >= min_conf
        )

        passed = count >= min_rep
        task_results.append(
            {
                "prompt": task["prompt"],
                "target_sign": target,
                "min_repetitions": min_rep,
                "min_confidence": min_conf,
                "matched_count": count,
                "passed": passed,
            }
        )

    overall_pass = all(t["passed"] for t in task_results)

    # Store summary in DB (for dashboard/progress)
    assessments_coll = db["assessments"]
    assessments_coll.insert_one(
        {
            "timestamp": now,
            "user_id": user_id,
            "lesson_id": num,
            "lesson_title": lesson_obj["title"],
            "tasks": task_results,
            "passed": overall_pass,
        }
    )

    return render_template(
        "assessment.html",
        lesson=lesson_obj,
        lesson_num=num,
        task_results=task_results,
        overall_pass=overall_pass,
    )


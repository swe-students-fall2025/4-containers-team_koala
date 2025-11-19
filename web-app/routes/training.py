"""Training routes for lessons and assessments in the ASL Trainer app."""

from __future__ import annotations

import time
from typing import List, Dict, Any

from bson import ObjectId
import requests  # pylint: disable=import-error
from flask import (
    Blueprint,
    redirect,
    render_template,
    session,
    url_for,
    current_app,
    request,
    jsonify,
)

training = Blueprint("training", __name__, url_prefix="/training")

# ----------------- LESSONS -----------------

LESSONS: List[Dict[str, Any]] = [
    {
        "id": 1,
        "title": "Lesson 1: ASL Alphabet A–G",
        "description": "Learn and practice ASL handshapes for the letters A through G.",
    },
    {
        "id": 2,
        "title": "Lesson 2: ASL Alphabet H–N",
        "description": "Learn and practice ASL handshapes for the letters H through N.",
    },
    {
        "id": 3,
        "title": "Lesson 3: ASL Alphabet O–U",
        "description": "Learn and practice ASL handshapes for the letters O through U.",
    },
    {
        "id": 4,
        "title": "Lesson 4: ASL Alphabet V–Z",
        "description": "Learn and practice ASL handshapes for the letters V through Z.",
    },
    {
        "id": 5,
        "title": "Final Practice Lesson",
        "description": "Review all letters A–Z and test your recognition skills.",
    },
]

LESSON_MAP = {lesson["id"]: lesson for lesson in LESSONS}

IMAGE_MAP = {
    1: "a_to_g.png",
    2: "h_to_n.png",
    3: "o_to_u.png",
    4: "v_to_z.png",
    5: "all_letters.png",
}

# ----------------- ASSESSMENTS -----------------

ASSESSMENTS: Dict[int, Dict[str, Any]] = {
    1: {
        "title": "Lesson 1 Assessment",
        "time_window_seconds": 60,
        "tasks": [
            {
                "prompt": "Sign the letter A three times.",
                "target_sign": "A",
                "min_repetitions": 3,
                "min_confidence": 0.6,
            },
            {
                "prompt": "Sign the letter C three times.",
                "target_sign": "C",
                "min_repetitions": 3,
                "min_confidence": 0.6,
            },
        ],
    },
    2: {
        "title": "Lesson 2 Assessment",
        "time_window_seconds": 60,
        "tasks": [
            {
                "prompt": "Sign the letter H three times.",
                "target_sign": "H",
                "min_repetitions": 3,
                "min_confidence": 0.6,
            },
            {
                "prompt": "Sign the letter L three times.",
                "target_sign": "L",
                "min_repetitions": 3,
                "min_confidence": 0.6,
            },
        ],
    },
    3: {
        "title": "Lesson 3 Assessment",
        "time_window_seconds": 60,
        "tasks": [
            {
                "prompt": "Sign the letter O three times.",
                "target_sign": "O",
                "min_repetitions": 3,
                "min_confidence": 0.6,
            },
            {
                "prompt": "Sign the letter R three times.",
                "target_sign": "R",
                "min_repetitions": 3,
                "min_confidence": 0.6,
            },
        ],
    },
    4: {
        "title": "Lesson 4 Assessment",
        "time_window_seconds": 60,
        "tasks": [
            {
                "prompt": "Sign the letter W three times.",
                "target_sign": "W",
                "min_repetitions": 3,
                "min_confidence": 0.6,
            },
            {
                "prompt": "Sign the letter Y three times.",
                "target_sign": "Y",
                "min_repetitions": 3,
                "min_confidence": 0.6,
            },
        ],
    },
    5: {
        "title": "Final Practice Assessment",
        "time_window_seconds": 90,
        "tasks": [
            {
                "prompt": "Sign the letter B three times.",
                "target_sign": "B",
                "min_repetitions": 3,
                "min_confidence": 0.6,
            },
            {
                "prompt": "Sign the letter R three times.",
                "target_sign": "R",
                "min_repetitions": 3,
                "min_confidence": 0.6,
            },
            {
                "prompt": "Sign the letter V three times.",
                "target_sign": "V",
                "min_repetitions": 3,
                "min_confidence": 0.6,
            },
        ],
    },
}

ML_API_URL = "http://ml:8080/predict"


# ---------------- ROUTES -----------------


@training.route("/")
def lessons() -> str:
    """Display all ASL lessons."""
    if "user_id" not in session:
        return redirect(url_for("auth.login"))
    return render_template("lessons.html", lessons=LESSONS)


@training.route("/lesson/<int:num>")
def lesson(num: int) -> str:
    """Display a single lesson page."""
    if "user_id" not in session:
        return redirect(url_for("auth.login"))

    lesson_obj = LESSON_MAP.get(num)
    if not lesson_obj:
        return redirect(url_for("training.lessons"))

    return render_template(
        "lesson.html",
        lesson_num=num,
        description=lesson_obj["description"],
        image_file=IMAGE_MAP.get(num, "all_letters.png"),
    )


def call_ml_api(points: list) -> tuple[str, float] | tuple[None, None]:
    """Call the ML API and return (letter, confidence)."""
    try:
        response = requests.post(ML_API_URL, json={"points": points}, timeout=5)
        data = response.json() if response.ok else {}
        letter = data.get("letter")
        confidence = float(data.get("confidence", 0.0))
        if not letter:
            return None, None
        return letter, confidence
    except Exception as exc:  # pylint: disable=broad-exception-caught
        print("ML API error:", exc)
        return None, None


def save_detection(db, user_id: str, lesson_id: int, letter: str, confidence: float) -> bool:
    """Save a detection to the database."""
    detection = {
        "user_id": user_id,
        "lesson_id": lesson_id,
        "timestamp": time.time(),
        "sign_label": letter,
        "confidence": confidence,
    }
    try:
        db["detections"].insert_one(detection)
        return True
    except Exception as exc:  # pylint: disable=broad-exception-caught
        print("DB error:", exc)
        return False


def check_tasks(db, user_id: str, lesson_id: int, assessment_def: dict) -> tuple[list[dict], bool]:
    """Check task completion and return task results and overall pass."""
    task_results = []
    window_start = time.time() - assessment_def["time_window_seconds"]

    for task in assessment_def["tasks"]:
        matched_count = db["detections"].count_documents(
            {
                "user_id": user_id,
                "lesson_id": lesson_id,
                "sign_label": task["target_sign"],
                "confidence": {"$gte": task["min_confidence"]},
                "timestamp": {"$gte": window_start},
            }
        )
        task_results.append(
            {
                "prompt": task["prompt"],
                "target_sign": task["target_sign"],
                "min_repetitions": task["min_repetitions"],
                "min_confidence": task["min_confidence"],
                "matched_count": matched_count,
                "passed": matched_count >= task["min_repetitions"],
            }
        )

    overall_pass = all(result["passed"] for result in task_results)
    return task_results, overall_pass


def update_progress(db, user_id: str, lesson_id: int, assessment_def: dict) -> None:
    """Update user progress if assessment passed."""
    assessment_title = assessment_def.get("title", f"Lesson {lesson_id} Assessment")
    db["users"].update_one(
        {"_id": ObjectId(user_id)},
        {
            "$addToSet": {
                "progress.assessments_taken": assessment_title,
                "progress.lessons_completed": lesson_id,
            }
        },
    )


@training.route("/lesson/<int:num>/assessment", methods=["GET", "POST"])
def assessment(num: int):
    """Handle lesson assessments and prediction scoring."""
    user_id = session.get("user_id")
    if not user_id:
        if request.method == "POST":
            return jsonify({"error": "Not logged in"}), 401
        return redirect(url_for("auth.login"))

    lesson_obj = LESSON_MAP.get(num)
    assessment_def = ASSESSMENTS.get(num)
    if not lesson_obj or not assessment_def:
        return redirect(url_for("training.lessons"))

    db = current_app.db

    if request.method == "POST":
        data = request.get_json(silent=True)
        points = data.get("points") if data else None

        if not points or len(points) != 21:
            return jsonify({"error": "Invalid landmarks"}), 400

        letter, confidence = call_ml_api(points)
        if not letter:
            return jsonify({"error": "Failed to get prediction"}), 500

        if not save_detection(db, user_id, num, letter, confidence):
            return jsonify({"error": "Failed to save detection"}), 500

        task_results, overall_pass = check_tasks(db, user_id, num, assessment_def)

        if overall_pass:
            update_progress(db, user_id, num, assessment_def)

        return jsonify(
            {
                "current_letter": letter,
                "current_confidence": confidence,
                "task_results": task_results,
                "overall_pass": overall_pass,
            }
        )

    return render_template(
        "assessment.html",
        lesson=lesson_obj,
        lesson_num=num,
        tasks=assessment_def["tasks"],
    )

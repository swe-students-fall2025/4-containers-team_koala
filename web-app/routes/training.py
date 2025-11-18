from __future__ import annotations

import time
from typing import List, Dict, Any
import requests
from flask import (
    Blueprint,
    redirect,
    render_template,
    session,
    url_for,
    current_app,
    request, 
    jsonify
)

training = Blueprint("training", __name__, url_prefix="/training")

LESSONS: List[Dict[str, Any]] = [
    {
        "id": 1,
        "title": "Lesson 1: ASL Alphabet A-G",
        "description": "Learn and practice ASL handshapes for the letters A through G.",
        # "signs": ["A", "B", "C", "D", "E", "F", "G"],
    },
    {
        "id": 2,
        "title": "Lesson 2: ASL Alphabet H-N",
        "description": "Learn and practice ASL handshapes for the letters H through N.",
        # "signs": ["H", "I", "J", "K", "L", "M", "N"],
    },
    {
        "id": 3,
        "title": "Lesson 3: ASL Alphabet O-U",
        "description": "Learn and practice ASL handshapes for the letters O through U.",
        # "signs": ["O", "P", "Q", "R", "S", "T", "U"],
    },
    {
        "id": 4,
        "title": "Lesson 4: ASL Alphabet V-Z",
        "description": "Learn and practice ASL handshapes for the letters V through Z.",
        # "signs": ["V", "W", "X", "Y", "Z"],
    },
    {
        "id": 5,
        "title": "Final Practice Lesson",
        "description": "Review all letters A-Z and test your recognition skills.",
        # "signs": list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
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
ML_API_URL = "http://ml-api:8080/predict"

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

    IMAGE_MAP = {
        1: "a_to_g.png",
        2: "h_to_n.png",
        3: "o_to_u.png",
        4: "v_to_z.png",
        5: "all_letters.png"
    }

    return render_template(
        "lesson.html",
        lesson_num=num,
        description=lesson_obj["description"],
        # signs=lesson_obj["signs"],
        image_file=IMAGE_MAP.get(num, "all_letters.png")
    )


@training.route("/lesson/<int:num>/assessment", methods=["GET", "POST"])
def assessment(num: int):
    """Run an assessment for a given lesson with ML API detection."""
    if "user_id" not in session:
        return redirect(url_for("auth.login"))

    user_id = session["user_id"]
    lesson_obj = LESSON_MAP.get(num)
    assessment_def = ASSESSMENTS.get(num)

    if lesson_obj is None or assessment_def is None:
        return redirect(url_for("training.lessons"))

    db = current_app.db

    # ---------- POST: process landmarks ----------
    if request.method == "POST":
        data = request.get_json()
        points = data.get("points")
        if not points or len(points) != 21:
            return jsonify({"error": "Invalid landmarks"}), 400

        # Call ML API
        try:
            ml_resp = requests.post(ML_API_URL, json={"points": points}, timeout=5).json()
            letter = ml_resp.get("letter")
            confidence = float(ml_resp.get("confidence", 0.0))
        except Exception:
            return jsonify({"error": "Failed to contact ML API"}), 500

        # Save detection
        detection = {
            "user_id": user_id,
            "lesson_id": num,
            "timestamp": time.time(),
            "sign_label": letter,
            "confidence": confidence,
        }
        db["detections"].insert_one(detection)

        # Check assessment tasks
        task_results = []
        for task in assessment_def["tasks"]:
            target = task["target_sign"]
            min_rep = task["min_repetitions"]
            min_conf = task["min_confidence"]

            window_start = time.time() - assessment_def["time_window_seconds"]
            count = db["detections"].count_documents({
                "user_id": user_id,
                "lesson_id": num,
                "sign_label": target,
                "confidence": {"$gte": min_conf},
                "timestamp": {"$gte": window_start}
            })

            passed = count >= min_rep
            task_results.append({
                "prompt": task["prompt"],
                "target_sign": target,
                "min_repetitions": min_rep,
                "min_confidence": min_conf,
                "matched_count": count,
                "passed": passed
            })

        overall_pass = all(t["passed"] for t in task_results)

        return jsonify({
            "current_letter": letter,
            "current_confidence": confidence,
            "task_results": task_results,
            "overall_pass": overall_pass
        })

    # ---------- GET: render HTML ----------
    return render_template(
        "assessment.html",
        lesson=lesson_obj,
        lesson_num=num,
        tasks=assessment_def["tasks"]
    )
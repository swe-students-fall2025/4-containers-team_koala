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
    {"id": 1, "title": "Lesson 1: ASL Alphabet A-G", "description": "Learn and practice ASL handshapes for the letters A through G."},
    {"id": 2, "title": "Lesson 2: ASL Alphabet H-N", "description": "Learn and practice ASL handshapes for the letters H through N."},
    {"id": 3, "title": "Lesson 3: ASL Alphabet O-U", "description": "Learn and practice ASL handshapes for the letters O through U."},
    {"id": 4, "title": "Lesson 4: ASL Alphabet V-Z", "description": "Learn and practice ASL handshapes for the letters V through Z."},
    {"id": 5, "title": "Final Practice Lesson", "description": "Review all letters A-Z and test your recognition skills."},
]

LESSON_MAP = {lesson["id"]: lesson for lesson in LESSONS}

ASSESSMENTS: Dict[int, Dict[str, Any]] = {
    1: {"time_window_seconds": 60, "tasks": [
        {"prompt": "Sign the letter A three times.", "target_sign": "A", "min_repetitions": 3, "min_confidence": 0.8},
        {"prompt": "Sign the letter C three times.", "target_sign": "C", "min_repetitions": 3, "min_confidence": 0.8}
    ]},
    2: {"time_window_seconds": 60, "tasks": [
        {"prompt": "Sign the letter H three times.", "target_sign": "H", "min_repetitions": 3, "min_confidence": 0.8},
        {"prompt": "Sign the letter N three times.", "target_sign": "N", "min_repetitions": 3, "min_confidence": 0.8}
    ]},
    3: {"time_window_seconds": 60, "tasks": [
        {"prompt": "Sign the letter O three times.", "target_sign": "O", "min_repetitions": 3, "min_confidence": 0.8},
        {"prompt": "Sign the letter S three times.", "target_sign": "S", "min_repetitions": 3, "min_confidence": 0.8}
    ]},
    4: {"time_window_seconds": 60, "tasks": [
        {"prompt": "Sign the letter V three times.", "target_sign": "V", "min_repetitions": 3, "min_confidence": 0.8},
        {"prompt": "Sign the letter Z three times.", "target_sign": "Z", "min_repetitions": 3, "min_confidence": 0.8}
    ]},
    5: {"time_window_seconds": 90, "tasks": [
        {"prompt": "Sign the letter A three times.", "target_sign": "A", "min_repetitions": 3, "min_confidence": 0.8},
        {"prompt": "Sign the letter M three times.", "target_sign": "M", "min_repetitions": 3, "min_confidence": 0.8},
        {"prompt": "Sign the letter Z three times.", "target_sign": "Z", "min_repetitions": 3, "min_confidence": 0.8}
    ]}
}

ML_API_URL = "http://ml:8080/predict"

# ---------------- ROUTES ----------------
@training.route("/")
def lessons() -> str:
    if "user_id" not in session:
        return redirect(url_for("auth.login"))
    return render_template("lessons.html", lessons=LESSONS)

@training.route("/lesson/<int:num>")
def lesson(num: int) -> str:
    if "user_id" not in session:
        return redirect(url_for("auth.login"))

    lesson_obj = LESSON_MAP.get(num)
    if not lesson_obj:
        return redirect(url_for("training.lessons"))

    IMAGE_MAP = {1: "a_to_g.png", 2: "h_to_n.png", 3: "o_to_u.png", 4: "v_to_z.png", 5: "all_letters.png"}

    return render_template(
        "lesson.html",
        lesson_num=num,
        description=lesson_obj["description"],
        image_file=IMAGE_MAP.get(num, "all_letters.png")
    )

@training.route("/lesson/<int:num>/assessment", methods=["GET", "POST"])
def assessment(num: int):
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

        # ML API call
        try:
            ml_resp_raw = requests.post(ML_API_URL, json={"points": points}, timeout=5)
            ml_resp = ml_resp_raw.json() if ml_resp_raw.ok else {}
            letter = ml_resp.get("letter")
            confidence = float(ml_resp.get("confidence", 0.0))
            if not letter:
                raise ValueError("No prediction returned")
        except Exception as e:
            print("ML API error:", e)
            return jsonify({"error": "Failed to get prediction"}), 500

        # Save detection
        detection = {"user_id": user_id, "lesson_id": num, "timestamp": time.time(),
                     "sign_label": letter, "confidence": confidence}
        try:
            db["detections"].insert_one(detection)
        except Exception as e:
            print("DB error:", e)
            return jsonify({"error": "Failed to save detection"}), 500

        # Check tasks
        task_results = []
        window_start = time.time() - assessment_def["time_window_seconds"]
        for task in assessment_def["tasks"]:
            count = db["detections"].count_documents({
                "user_id": user_id,
                "lesson_id": num,
                "sign_label": task["target_sign"],
                "confidence": {"$gte": task["min_confidence"]},
                "timestamp": {"$gte": window_start}
            })
            task_results.append({
                "prompt": task["prompt"],
                "target_sign": task["target_sign"],
                "min_repetitions": task["min_repetitions"],
                "min_confidence": task["min_confidence"],
                "matched_count": count,
                "passed": count >= task["min_repetitions"]
            })

        overall_pass = all(t["passed"] for t in task_results)

        return jsonify({
            "current_letter": letter,
            "current_confidence": confidence,
            "task_results": task_results,
            "overall_pass": overall_pass
        })

    # GET request
    return render_template("assessment.html", lesson=lesson_obj, lesson_num=num, tasks=assessment_def["tasks"])

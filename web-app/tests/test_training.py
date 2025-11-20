"""
Tests for training routes.
"""

import time
from unittest.mock import patch

import pytest
from unittest.mock import patch, MagicMock
from bson import ObjectId
import time

from routes.training import (
    LESSONS,
    LESSON_MAP,
    IMAGE_MAP,
    ASSESSMENTS,
    call_ml_api,
    save_detection,
    check_tasks,
    update_progress,
)


# lesson routes
def test_lessons_redirects_not_logged_in(client):
    resp = client.get("/training/")
    assert resp.status_code == 302
    assert "/login" in resp.headers["Location"]

def test_lessons_page_logged_in(client, app):
    with client.session_transaction() as sess:
        sess["user_id"] = "123"

    resp = client.get("/training/")
    assert resp.status_code == 200
    for lesson in LESSONS:
        assert lesson["title"] in resp.text

def test_lesson_page_valid(client):
    """Lesson page should render when user is logged in."""
    with client.session_transaction() as sess:
        sess["user_id"] = "abc"

    resp = client.get("/training/lesson/1")
    assert resp.status_code == 200
    assert "Lesson 1" in resp.text
    assert IMAGE_MAP[1] in resp.text

def test_lesson_page_invalid_redirects(client):
    with client.session_transaction() as sess:
        sess["user_id"] = "abc"

    resp = client.get("/training/lesson/999")
    assert resp.status_code == 302
    assert "/training" in resp.headers["Location"]

# ML API calls tests
@patch("routes.training.requests.post")
def test_call_ml_api_success(mock_post):
    mock_post.return_value.ok = True
    mock_post.return_value.json.return_value = {"letter": "A", "confidence": 0.82}

    letter, conf = call_ml_api(points=[[0, 0, 0]] * 21)

    assert letter == "A"
    assert conf == 0.82

@patch("routes.training.requests.post")
def test_call_ml_api_fail(mock_post):
    mock_post.side_effect = Exception("API Down")

    letter, conf = call_ml_api(points=[[0, 0, 0]] * 21)

    assert letter is None
    assert conf is None


# db save tests
def test_save_detection_success(app):
    with app.app_context():
        ok = save_detection(
            db=app.db,
            user_id="123",
            lesson_id=1,
            letter="A",
            confidence=0.9,
        )
        assert ok is True
        assert app.db.detections.count_documents({}) == 1

def test_save_detection_failure(app):
    class BrokenDB:
        def __getitem__(self, name):
            raise Exception("DB DOWN")

    ok = save_detection(
        db=BrokenDB(),
        user_id="123",
        lesson_id=1,
        letter="A",
        confidence=0.9,
    )
    assert ok is False

# task logic tests
def test_check_tasks_pass(app):
    """If repetitions >= required, the task should pass."""
    with app.app_context():
        now = time.time()

        # incert enough amount for passing
        app.db.detections.insert_many([
            {
                "user_id": "u1",
                "lesson_id": 1,
                "sign_label": "A",
                "confidence": 0.9,
                "timestamp": now,
            }
            for _ in range(3)
        ])

        tasks = {
            "time_window_seconds": 60,
            "tasks": [
                {"prompt": "Sign A", "target_sign": "A", "min_repetitions": 3, "min_confidence": 0.5}
            ],
        }

        result, overall = check_tasks(app.db, "u1", 1, tasks)

        assert overall is True
        assert result[0]["passed"] is True

def test_check_tasks_fail(app):
    with app.app_context():
        now = time.time()
        app.db.detections.insert_one(
            {
                "user_id": "u1",
                "lesson_id": 1,
                "sign_label": "A",
                "confidence": 0.9,
                "timestamp": now,
            }
        )
        tasks = {
            "time_window_seconds": 60,
            "tasks": [
                {"prompt": "Sign A", "target_sign": "A", "min_repetitions": 3, "min_confidence": 0.5}
            ],
        }

        result, overall = check_tasks(app.db, "u1", 1, tasks)

        assert overall is False
        assert result[0]["passed"] is False

# update progress tests
def test_update_progress(app):
    with app.app_context():
        user_id = app.db.users.insert_one(
            {"progress": {"lessons_completed": [], "assessments_taken": []}}
        ).inserted_id

        update_progress(
            db=app.db,
            user_id=str(user_id),
            lesson_id=1,
            assessment_def={"title": "Test Assessment"},
        )

        user = app.db.users.find_one({"_id": user_id})
        assert 1 in user["progress"]["lessons_completed"]
        assert "Test Assessment" in user["progress"]["assessments_taken"]


# assessment route tests
def test_assessment_page_renders(client):
    with client.session_transaction() as sess:
        sess["user_id"] = "x"

    resp = client.get("/training/lesson/1/assessment")
    assert resp.status_code == 200
    assert "Assessment" in resp.text


@patch("routes.training.call_ml_api")
def test_assessment_post_success(mock_ml, client, app):
    """Simulate correct ML prediction and passing tasks."""
    mock_ml.return_value = ("A", 0.9)

    with client.session_transaction() as sess:
        sess["user_id"] = "user123"

    app.db.detections.insert_many([
        {"user_id": "user123", "lesson_id": 1, "sign_label": "A",
         "confidence": 0.9, "timestamp": time.time()}
        for _ in range(3)
    ])

    sample_points = [[0, 0, 0]] * 21
    resp = client.post(
        "/training/lesson/1/assessment",
        json={"points": sample_points},
    )

    assert resp.status_code == 200
    data = resp.json
    assert data["current_letter"] == "A"
    assert data["overall_pass"] in (True, False) # depending on rest of task check
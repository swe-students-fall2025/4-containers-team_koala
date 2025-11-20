"""
Tests for authentication routes.
"""

from datetime import datetime
from werkzeug.security import generate_password_hash

from routes.auth import validate_registration, get_user_if_valid


def test_validate_registration_success(app) -> None:
    """Registration succeeds with valid input."""
    with app.app_context():
        ok, msg = validate_registration(
            email="test@example.com",
            username="tester",
            password="secret123",
        )
        assert ok is True
        assert msg == ""


def test_validate_registration_rejects_empty_fields(app) -> None:
    """Empty email/username/password should be rejected."""
    with app.app_context():
        ok, msg = validate_registration(email="", username="", password="")
        assert ok is False
        assert "cannot be empty" in msg


def test_validate_registration_rejects_short_password(app) -> None:
    """Password shorter than 6 chars should be rejected."""
    with app.app_context():
        ok, msg = validate_registration(
            email="short@example.com",
            username="shortuser",
            password="123",
        )
        assert ok is False
        assert "at least 6 characters" in msg


def test_validate_registration_rejects_existing_username(app) -> None:
    """Should not allow duplicate usernames."""
    with app.app_context():
        _ = app.db.users.insert_one(
            {
                "username": "duplicate",
                "email": "dup@example.com",
                "password_hash": generate_password_hash("secret123"),
                "created_at": datetime.utcnow(),
            }
        ).inserted_id

        ok, msg = validate_registration(
            email="new@example.com",
            username="duplicate",
            password="secret123",
        )
        assert ok is False
        assert "Username already exists" in msg


def test_validate_registration_rejects_existing_email(app) -> None:
    """Should not allow duplicate emails."""
    with app.app_context():
        _ = app.db.users.insert_one(
            {
                "username": "someone",
                "email": "taken@example.com",
                "password_hash": generate_password_hash("secret123"),
                "created_at": datetime.utcnow(),
            }
        ).inserted_id

        ok, msg = validate_registration(
            email="taken@example.com",
            username="newuser",
            password="secret123",
        )
        assert ok is False
        assert "Email already registered" in msg


def test_get_user_if_valid_returns_user_for_correct_password(app) -> None:
    """get_user_if_valid should return the user document on success."""
    with app.app_context():
        users = app.db.users
        _ = users.insert_one(
            {
                "username": "alice",
                "email": "alice@example.com",
                "password_hash": generate_password_hash("secret123"),
                "created_at": datetime.utcnow(),
                "last_login": None,
                "progress": {"lessons_completed": [], "assessments_taken": []},
            }
        ).inserted_id

        user = get_user_if_valid("alice", "secret123")
        assert user is not None
        assert user["username"] == "alice"


def test_get_user_if_valid_returns_none_for_wrong_password(app) -> None:
    """get_user_if_valid should return None when password is wrong."""
    with app.app_context():
        users = app.db.users
        _ = users.insert_one(
            {
                "username": "bob",
                "email": "bob@example.com",
                "password_hash": generate_password_hash("correctpw"),
                "created_at": datetime.utcnow(),
            }
        ).inserted_id

        user = get_user_if_valid("bob", "wrongpw")
        assert user is None


def test_login_route_success(client, app) -> None:
    """POST /login should log the user in with valid credentials."""
    with app.app_context():
        users = app.db.users
        _ = users.insert_one(
            {
                "username": "dora",
                "email": "dora@example.com",
                "password_hash": generate_password_hash("secret123"),
                "created_at": datetime.utcnow(),
                "last_login": None,
                "progress": {"lessons_completed": [], "assessments_taken": []},
            }
        ).inserted_id

    resp = client.post(
        "/login",
        data={"username": "dora", "password": "secret123"},
        follow_redirects=False,
    )

    # Should redirect to dashboard.home
    assert resp.status_code == 302
    assert "/dashboard" in resp.headers["Location"] or "dashboard" in resp.headers["Location"]

    # Check session
    with client.session_transaction() as sess:
        assert sess.get("user_id") is not None
        assert sess.get("username") == "dora"

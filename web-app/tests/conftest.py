"""
Pytest fixtures for web-app tests.
"""

import os
import sys

# Ensure web_app package is in PYTHONPATH
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE_DIR)  

import pytest
import mongomock
from web_app import create_app  


@pytest.fixture
def app():
    """
    Create a Flask app instance for testing.

    Sets TESTING=True, disables CSRF, and attaches a mongomock database.
    """
    app_instance = create_app()
    app_instance.config["TESTING"] = True
    app_instance.config["WTF_CSRF_ENABLED"] = False

    client = mongomock.MongoClient()
    app_instance.db = client["test_db"]

    yield app_instance

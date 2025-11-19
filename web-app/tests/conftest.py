import pytest
import mongomock

import os
import sys

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE_DIR)
from __init__ import create_app

@pytest.fixture
def app():
    app = create_app()
    app.config["TESTING"] = True
    app.config["WTF_CSRF_ENABLED"] = False

    client = mongomock.MongoClient()
    app.db = client["test_db"]

    yield app
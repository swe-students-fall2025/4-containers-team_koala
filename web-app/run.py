import sys
import os

# Add the web-app folder to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "web-app"))

# Import the create_app factory
from __init__ import create_app

app = create_app()

if __name__ == "__main__":
    # Enable debug mode for auto-reload
    app.run(host="0.0.0.0", port=5000, debug=True)

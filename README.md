[![ML Client CI](https://github.com/swe-students-fall2025/4-containers-team_koala/actions/workflows/ml-client-ci.yml/badge.svg)](https://github.com/swe-students-fall2025/4-containers-team_koala/actions/workflows/ml-client-ci.yml)
[![Web App CI](https://github.com/swe-students-fall2025/4-containers-team_koala/actions/workflows/web-app-ci.yml/badge.svg)](https://github.com/swe-students-fall2025/4-containers-team_koala/actions/workflows/web-app-ci.yml)

---

# ASL Learning App – Containerized Machine Learning System

This project is a **containerized ASL (American Sign Language) learning application**.

Using a **webcam**, a **machine learning client** detects ASL letters in real time and stores these detections in a **MongoDB** database. A **Flask web app** then uses that data to:

- Teach ASL letters through **lessons**
- Evaluate learners through **assessments** that use real ML detections
- Show **progress dashboards** over time

Everything runs as **three cooperating Docker containers**:

1. **Machine Learning Client** (Python, OpenCV/MediaPipe, `pymongo`)  
2. **Web App** (Python, Flask, Jinja2, `pymongo`)  
3. **MongoDB Database** (official `mongo` image)

The repository is a **monorepo**: both Python subsystems live here and share a single `docker-compose.yml`.

---

## Team

- [Kylie Lin](https://github.com/kylin1209)  
- [Alex Xie](https://github.com/axie22)  
- [Jacob Ng](https://github.com/jng20)
- [Jordan Lee](https://github.com/jjl9930)  
- [Phoebe Huang](https://github.com/phoebelh)   

---

## Overview

- You start the system with **Docker Compose**.
- The **MongoDB** database starts first.
- The **ML client** container connects to your **webcam**, looks at your hand, and tries to recognize which ASL letter you are signing (A–Z).
- For every recognized letter, it saves a small record into MongoDB: *who* signed it, *what* letter, *how confident* the model is, and *when* it happened.
- The **web app** lets you:
  - Log in
  - Browse **lessons** (e.g., “Lesson 1: A–G”)
  - Run **assessments**, where you’re asked to sign certain letters several times
  - See a **pass/fail** result based on what the ML client actually saw
  - View a **dashboard** of which lessons you’ve passed so far

You can think of it as a **small, interactive ASL course** that uses your webcam and a simple ML model to check your work.

---

## Repository Structure

```text
.
├── machine-learning-client/
│   ├── Pipfile / Pipfile.lock       # Pipenv environment for ML client
│   ├── src/
│   │   ├── webcam_client.py         # main loop: webcam → ML model → MongoDB
│   │   ├── model.py                 # Landmark-based ASL classifier
│   │   ├── landmarks.py             # MediaPipe / preprocessing utilities
│   │   └── db_client.py             # MongoDB detection writer
│   ├── tests/                       # pytest tests for ML client
│   ├── Dockerfile
│   └── .env.example                 # example env file for this subsystem
│
├── web-app/
│   ├── Pipfile / Pipfile.lock       # Pipenv environment for web app
│   ├── app.py                       # Flask app factory (create_app)
│   ├── run.py                       # entrypoint (used by Docker CMD)
│   ├── routes/
│   │   ├── auth.py                  # login/registration
│   │   ├── training.py              # lessons + assessments
│   │   └── dashboard.py             # user progress dashboard
│   ├── templates/                   # Jinja2 templates
│   ├── static/                      # CSS / JS / images
│   ├── tests/                       # pytest + pytest-flask tests
│   ├── Dockerfile
│   └── .env.example                 # example env file for this subsystem
│
├── docker-compose.yml               # orchestrates mongodb + ml + web
├── README.md                        # this file
├── .env.example                     # root env example (for Docker Compose)
├── .gitignore
└── .github/
    └── workflows/
        ├── ml-client-ci.yml         # build + test ML client
        └── web-app-ci.yml           # build + test web app

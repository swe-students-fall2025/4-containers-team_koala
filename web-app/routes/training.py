from flask import Blueprint, render_template

training_bp = Blueprint("training", __name__)

@training_bp.get("/training")
def training():
    return render_template("lesson.html")

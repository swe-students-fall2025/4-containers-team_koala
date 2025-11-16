from flask import Blueprint, render_template

auth_bp = Blueprint("auth", __name__)

@auth_bp.get("/login")
def login():
    return render_template("login.html")

@auth_bp.get("/register")
def register():
    return render_template("register.html")

from flask import Blueprint, render_template, request, redirect, url_for, flash, make_response, jsonify, send_file

import os
from datetime import datetime
from sqlalchemy import func
main = Blueprint('main', __name__)

@main.route("/")
def home():
    imgURL = "https://25.media.tumblr.com/tumblr_m9lp7aUZps1qb2cp4o1_500.gif"
    
    return render_template('index.html', image_url=imgURL)


# @main.route("/login", methods=["GET", "POST"])
# def login():
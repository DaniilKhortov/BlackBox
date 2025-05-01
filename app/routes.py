from flask import Blueprint, render_template, request, redirect, url_for, flash, make_response, jsonify, send_file

import os
from datetime import datetime
from sqlalchemy import func
main = Blueprint('main', __name__)

@main.route("/")
def home():
    imgURL = "https://25.media.tumblr.com/tumblr_m9lp7aUZps1qb2cp4o1_500.gif"
    
    return render_template('index.html', image_url=imgURL)

@main.route("/decrypt", methods=["POST"])
def decrypt():
    data = request.get_json()

    text = data.get("text", "") if data else ""
    

    #Тут надсилається тип шифру та розшифрований текст на сервер
    result = {
        "Type": "atbash",
        "Result" : text
    }
    
    # result = {
    #     "Type": "caesar",
    #     "Result" : text
    # }
    # result = {
    #     "Type": "pl",
    #     "Result" : text
    # } 
    
    #Це вивід помилки на сайті
    # result = {
    #     "Type": "err",
    #     "Result" : "Error!"
    # }
    
    return jsonify({"result": result})


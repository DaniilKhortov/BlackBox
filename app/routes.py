from flask import Blueprint, render_template, request, redirect, url_for, flash, make_response, jsonify, send_file


import os
from datetime import datetime
from sqlalchemy import func
main = Blueprint('main', __name__)

basedir = os.path.abspath(os.path.dirname(__file__))
from .complete.Classificator import decrypt_sentence_interface, initialize_system

initialize_system()

@main.route("/")
def home():
    imgURL = "https://25.media.tumblr.com/tumblr_m9lp7aUZps1qb2cp4o1_500.gif"
    
    return render_template('index.html', image_url=imgURL)

@main.route("/decrypt", methods=["POST"])
def decrypt():
    data = request.get_json()

    text = data.get("text", "") if data else ""
    
    initialize_system()
    cipher_type, result_s, msgs = decrypt_sentence_interface(text)
    print(f"\n--- Вхід: '{text}' ---")
    print(f"Тип шифру: {cipher_type}")
    print(f"Результат: '{result_s}'")
    print("Повідомлення:")
    for msg in msgs:
        print(f"  - {msg}")
    print("-" * 60)

    #Тут надсилається тип шифру та розшифрований текст на сервер
    result = {
          "Type": cipher_type,
          "Result" : result_s
        }
    # result = {
    #     "Type": "pl",
    #     "Result" : text
    # } 
    #Це вивід помилки на сайті
    if cipher_type == None:
        result = {
         "Type": "err",
         "Result" : msgs
     }
     
    
    return jsonify({"result": result})


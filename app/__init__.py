from flask import Flask
import os

basedir = os.path.abspath(os.path.dirname(__file__))


def decrypter_app():
    app = Flask(__name__)

    
    from .routes import main
    app.register_blueprint(main)

    return app
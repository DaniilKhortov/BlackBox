# Black box code breaker
<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li>
      <a href="#application-launch">Application launch </a>
    </li>
    <li>
      <a href="#project-structure">Project Structure</a>
    </li>
    <li>
      <a href="#database">Database</a>
    </li>
  </ol>
</details>

## About The Project
The purpose of this project is to demonstrate machine learning–based code decryption. Unlike brute force methods, ML-based decryption recognizes linguistic patterns, making it more efficient.

### Built With
Languages:
* [![JavaScript](https://img.shields.io/badge/JavaScript-F7DF1E?logo=javascript&logoColor=000)](#)
*	[![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=fff)](#)
* [![HTML](https://img.shields.io/badge/HTML-%23E34F26.svg?logo=html5&logoColor=white)](#)
* [![CSS](https://img.shields.io/badge/CSS-1572B6?logo=css3&logoColor=fff)](#)

Frameworks:
* [![Flask](https://img.shields.io/badge/Flask-000?logo=flask&logoColor=fff)](#)

Databases:
* [![MongoDB](https://img.shields.io/badge/MongoDB-4EA94B?logo=mongodb&logoColor=fff)](#)

## Getting Started

### Prerequisites
* Make sure MongoDB service is running locally or remotely before launching the app.
* Flask
  ```sh
  pip install Flask
  ```
* Scikit-learn
  ```sh
   pip install scikit-learn 
  ```
* Numpy
  ```sh
   pip install numpy
  ```

### Installation
1. Clone the repository
  ```sh
  git clone https://github.com/DaniilKhortov/file_exchanger_KhortovDP.git
  ```
2. Install libraries
  ```sh
  pip install pymongo
  ```
3.Change the git remote (optional if you want to change your origin)
  ```sh
  git remote set-url origin https://github.com/DaniilKhortov/file_exchanger_KhortovDP.git
  ```


## Application launch 
1. Run file run.py.
  ```sh
  python run.py
  ```
2. If everything is set-up correctly, Flask will start a local development server at http://127.0.0.1:8000 (or specified port).



## Project Structure
```bash
< PROJECT ROOT >
   |
   |-- app/                                # Core application directory
   |    |
   |    | -- routes.py                     # Main functions to work with client
   |    | -- __init__.py                   # Initialization of flask app, connection to database
   |    |
   |    |-- complete/                      # Stores working files
   |    |    |
   |    |    |-- Atbash.py                 # Decryptor for Atbash Language
   |    |    |-- Caesar.py                 # Decryptor for Caesar Language
   |    |    |-- Classificator.py          # ML module. Decides the type of inputed encryption method
   |    |    |-- Pl.py                     # Decryptor for Pig Latin Language
   |    |    |-- click_me.py               # Util, needed for decryptors
   |    |    |-- main.py                   # Debug util
   |    |
   |    |-- datasetsBackUP/                 # Stores dataset
   |    |    |
   |    |    |-- data.csv                   # Dataset for Caesar language. Uses real words
   |    |    |-- dataA.csv                  # Dataset for Atbash language. Uses real words
   |    |    |-- dataPL.csv                 # Dataset for Pig Latin language. Uses real words
   |    |    |-- EnigmaticCodes.Atbash.csv  # Dataset for Atbash language. Uses generated words. Not effective
   |    |    |-- EnigmaticCodes.Caesar.csv  # Dataset for Caesar language. Uses generated words. Not effective
   |    |    |-- EnigmaticCodes.PL.csv      # Dataset for Pig Latin language. Uses generated words. Not effective
   |    |    |
   |    |-- static/
   |    |    |
   |    |    |-- icon.ico 
   |    |    |-- css/
   |    |    |    |            
   |    |    |    |-- desktop.css          # Used by pages, provides adaptability of interface to the bigger screens
   |    |    |    |-- style.css            # Used by pages, responsible  for design 
   |    |    |-- js/                  
   |    |    |    |
   |    |    |    |-- index.js              # Responsible for datatransfer between client and server
   |    |    |
   |    |-- templates/
   |    |    |
   |    |    |-- index.html                # Main page
   |    |
   |-- run.py                              # Entry point for running Flask server
   |-- text.txt                            # Source text used for dataset generation.  Currently it has several chapters from "All Quiet on Western Front"
```

## Database
Software uses MongoDB. Collections must look like this:
```bash
< EnigmaticCodes >
   |
   |-- Atbash/
   |
   |-- Caesar/
   |
   |-- PL/
   |
```
Every collection is a dictionary with encrypted word, decrypted word and keys.
Datasets are located at BlackBox/app/datasetsBackUp. Caesar, Atbash and Pig Latin are named under data, dataA and dataPL accordingly.


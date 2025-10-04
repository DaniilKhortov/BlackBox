# Black box code breaker
(Description still in progress)
# File Exchanger
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
      <a href="#usage">Usage</a>
    </li>
    <li>
      <a href="#project-structure">Project Structure</a>
    </li>
    <li>
      <a href="#database">Database</a>
      <ul>
        <li><a href="#table-structures">Table Structures</a></li>
      </ul>
    </li>
    <li>
      <a href="#server-functions">Server Functions</a>
    </li>
    <li>
      <a href="#contact">Contact</a>
    </li>
  </ol>
</details>

## About The Project
The purpose of project is to show an ability of code decryption with machine learning tools. Unlike Brute Force method, decription with ML uses patterns in dictionaries, which is more efficient.

### Built With
Languages:
* [![JavaScript](https://img.shields.io/badge/JavaScript-F7DF1E?logo=javascript&logoColor=000)](#)
*	[![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=fff)](#)
* [![HTML](https://img.shields.io/badge/HTML-%23E34F26.svg?logo=html5&logoColor=white)](#)
* [![CSS](https://img.shields.io/badge/CSS-1572B6?logo=css3&logoColor=fff)](#)

Frameworks:
* [![Flask](https://img.shields.io/badge/Flask-000?logo=flask&logoColor=fff)](#)

Databases:
* ![MongoDB](https://img.shields.io/badge/MongoDB-%234ea94b.svg?style=for-the-badge&logo=mongodb&logoColor=white)

## Getting Started
### Prerequisites
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
2. If everything is set-up correctly, Flask will use device as a locale server and run app on http://127.0.0.1:8000


## Project Structure
```bash
< PROJECT ROOT >
   |
   |-- app/
   |    |
   |    | -- models.py                     # Database Tables
   |    | -- routes.py                     # Main functions to work with client
   |    | -- utils.py                      # Helpers to manipulate date, files  
   |    | -- __init__.py                   # Initialization of flask app, connection to database
   |    | -- accountsData.db               # Database
   |    |
   |    |-- storage/                       # Stores files that were sent to server
   |    |
   |    |-- static/
   |    |    |-- css/                  
   |    |    |    |
   |    |    |    |-- desktop.css          # Used by pages, provides adaptability of interface to the bigger screens
   |    |    |    |-- reset.css            # Used by pages, sets basic html-elements parameters to 0
   |    |    |    |-- style.css            # Used by pages, responsible  for design 
   |    |    |-- js/                  
   |    |    |    |
   |    |    |    |-- auth.js              # Responsible for keeping user authorized after closing window
   |    |    |    |-- download.js          # Responsible for downloading files by user and admin
   |    |    |    |-- elementsUtil.js      # Responsible for file upload and configuration by admin
   |    |
   |    |-- templates/
   |    |    |    
   |    |    |-- index.html                # Main page
   |    |    |-- login.html                # Authorization page
   |    |    |-- register.html             # Registration page
   |    |    |-- admin.html                # Modified main page for admin
   |    |    |
   |    |    |
   |    |    |
   |
   |-- run.py                              # Starts the app 
   |
   |-- ************************************************************************
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
   |-- ************************************************************************
```
Every collection is a dictionary with encrypted word, decrypted word and keys.
Datasets are located at BlackBox/app/datasetsBackUp. Caesar, Atbash and Pig Latin are named under data, dataA and dataPL accordingly.


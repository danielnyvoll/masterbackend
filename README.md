# Robot Soccer Simulation - Project Documentation

## Introduction

This guide provides instructions for setting up and running the Robot Soccer Simulation, which utilizes a Flask-SocketIO server to manage a game environment where robots, controlled by DQN agents, play soccer.

## Prerequisites

Before proceeding, ensure you have the following installed:
- Python 3.8 or higher
- pip (Python package installer)

## Installation

Clone the repository and install the required dependencies:

    git clone https://github.com/danielnyvoll/masterbackend.git

After the files are downloaded open the directory where you cloned the repo.

    cd masterbackend

When you are in the right directory run:

    pip install -r requirements.txt


## Running the Application

To start the Flask application, use the following command:

    python app.py

or

    flask run


This will initiate the Flask server with SocketIO on port 5000, accessible via http://localhost:5000.


When the application is running you can go to the next step:

## Setting Up the Frontend

Clone and set up the frontend application from its repository:

https://github.com/danielnyvoll/masterfrontend.git


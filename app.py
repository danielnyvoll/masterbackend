"""Flask application instance."""

from flask import Flask

app = Flask(__name__)

@app.route("/")
def hello_world():
    """Return a 'Hello, World!' message."""
    return "<p>Hello, World!</p>"

if __name__ == "__main__":
    app.run()

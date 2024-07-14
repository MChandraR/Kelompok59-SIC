import tensorflow as tf
import json
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def main():
    return render_template("index.html")



if __name__ == "__main__":
    app.run(debug=True)

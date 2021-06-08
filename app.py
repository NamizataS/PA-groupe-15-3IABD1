from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/result")
def resultPage():
    return render_template("result.html")


if __name__ == "__main__":
    app.run()
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/result")
def resultPage():
    return render_template("result.html")


@app.route('/uploader', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        f.save(secure_filename(f.filename))
        return render_template("result.html")

if __name__ == "__main__":
    app.run()
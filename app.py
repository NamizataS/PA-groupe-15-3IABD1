import os.path

from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import predict

app = Flask(__name__)
UPLOAD_FOLDER = os.path.join('static', 'img')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route("/")
def home():
    return render_template("index.html")


# @app.route("/result")
# def result_page_mlp():
# return render_template("result.html")


@app.route('/uploader_rbf', methods=['GET', 'POST'])
def upload_file_rbf():
    if request.method == 'POST':
        image = request.files['file']
        image.save(secure_filename(image.filename))
        image = secure_filename(image.filename)
        return redirect(url_for("display_results_rbf", image=image))
    else:
        return render_template("predict_rbf.html")


@app.route('/uploader_mlp', methods=['GET', 'POST'])
def upload_file_mlp():
    if request.method == 'POST':
        image = request.files['file']
        image.save(secure_filename(image.filename))
        image = secure_filename(image.filename)
        return redirect(url_for("display_results_mlp", image=image))
    else:
        return render_template("result.html")


@app.route('/display_results_rbf<image>')
def display_results_rbf(image):
    predict_obj = predict.Predict(image, (8, 8))
    prediction = predict_obj.predict_rbf()
    return render_template("display_results.html", pred=prediction, length=len(prediction))


@app.route('/display_results_mlp<image>')
def display_results_mlp(image):
    predict_obj = predict.Predict(image, (8, 8))
    prediction = predict_obj.predict_mlp()
    return render_template("display_results.html", pred=prediction, length=len(prediction))


if __name__ == "__main__":
    app.run(port=5000)

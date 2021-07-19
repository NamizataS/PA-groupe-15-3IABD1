import os.path

from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import predict

app = Flask(__name__)
UPLOAD_FOLDER = os.path.join('static','img')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/result")
def resultPage():
    return render_template("result.html")


@app.route('/uploader', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        image = request.files['file']
        #image_name = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
        image.save(secure_filename(image.filename))
        #image.save(image_name)
        image = secure_filename(image.filename)
        return redirect(url_for("display_results", image=image))
    else:
        return render_template("result.html")


@app.route('/display_results<image>')
def display_results(image):
    predict_obj = predict.Predict(image, (32, 32))
    prediction = predict_obj.predict_mlp()
    return render_template("display_results.html", pred=prediction, src_image=image, length=len(prediction))


if __name__ == "__main__":
    app.run(port=5000)

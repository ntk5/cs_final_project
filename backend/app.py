import json
import os
import uuid

import requests
import tensorflow as tf
from PIL import Image
from flask import Flask, send_from_directory, request, render_template, redirect, send_file
from flask_pymongo import PyMongo

from utils import allowed_file

app = Flask(__name__)
UPLOAD_FOLDER = 'data'
app.config['UPLOAD_FOLDER'] = os.path.join(app.root_path, UPLOAD_FOLDER)

try:
    if os.environ['IS_RUN_FROM_CONTAINER'] == 'yes':
        app.logger.debug("Running from docker")
        app.config["MONGO_URI"] = 'mongodb://' + os.environ['MONGODB_HOSTNAME'] + ':27017/' + os.environ[
            'MONGODB_DATABASE']
        app.config['INFERENCE_SERVICE'] = os.environ['INFERENCE_NAME']

except KeyError:
    app.logger.debug("Running from localhost")
    app.config["MONGO_URI"] = 'mongodb://localhost:27017/db'
    app.config['INFERENCE_SERVICE'] = 'localhost'

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.mkdir(app.config['UPLOAD_FOLDER'])

mongo = PyMongo(app)
db = mongo.db


@app.route('/')
def index():
    data = []
    res = db.skin_lesion_db.find()
    for item in res:
        data.append(dict(item))
    return render_template('index.html', data=data)


@app.route('/favicon.ico')
def get_favicon():
    return send_from_directory(os.path.join(app.root_path, 'templates'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')


@app.route('/analyze', methods=['POST'])
def analyze_image():
    request_fields = dict(request.form)
    file_storage_full_path = os.path.join(app.config['UPLOAD_FOLDER'], request_fields["image_path"])

    image = Image.open(file_storage_full_path)
    image = image.resize((224, 224))
    image = tf.keras.preprocessing.image.img_to_array(image).reshape(1, 224, 224, 3).astype('float32') / 255.0
    data = json.dumps({"signature_name": "serving_default", "instances": image.tolist()})
    r = requests.post(f"http://{app.config['INFERENCE_SERVICE']}:8501/v1/models/my_model:predict", data=data)

    response = float(dict(json.loads(r.content))['predictions'][0][0]) * 100
    request_fields["prognosis"] = response
    return str(response)


@app.route('/upload', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            return redirect(f"/analyze/")
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            return redirect(f"/analyze/")
        if file and allowed_file(file.filename):
            image_id = handle_image(file)
            return redirect(f"/analyze/{image_id}")
        return redirect(f"/analyze/")
    return render_template('upload.html')


def handle_image(file):
    image_id = str(uuid.uuid4())
    file_name = file.filename
    file_path = f"{image_id}_{file_name}"
    request_fields = dict(request.form)
    request_fields["id"] = image_id
    request_fields["file_name"] = file_name
    request_fields["image_path"] = file_path
    file_storage_full_path = os.path.join(app.config['UPLOAD_FOLDER'], file_path)
    file.save(file_storage_full_path)

    db.skin_lesion_db.insert_one(request_fields)
    return image_id


@app.route('/analyze/<case_id>')
def get_case(case_id):
    return get_search_page(case_id, "analyze.html")


@app.route('/search')
def search():
    request_id = request.args.get('search')
    return get_search_page(request_id)


def get_search_page(request_id, page_name='search.html'):
    res = []
    dat = db.skin_lesion_db.find({"id": request_id})
    for item in dat:
        res.append(dict(item))

    if len(res) > 0:
        return render_template(page_name, data=res)
    dat = db.skin_lesion_db.find({"patient_id": request_id})
    for item in dat:
        res.append(dict(item))
    return render_template(page_name, data=res)


@app.route('/static/<file_name>')
def get_static(file_name):
    return render_template(file_name)


@app.route('/background/<file_name>')
def get_background_image(file_name):
    return send_file(os.path.join("templates", file_name), mimetype='image/gif')


@app.route('/data/<file_name>')
def get_image(file_name):
    return send_file(os.path.join("data", file_name), mimetype='image/gif')


if __name__ == '__main__':
    app.run(debug=True)

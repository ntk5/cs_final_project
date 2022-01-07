ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


"""
running the inference service:
docker run -d -p 8501:8501
 --mount type=bind,
 source="new_model_data",
 target=/models/my_model/1 
 -e MODEL_NAME=my_model 
 -t tensorflow/serving
"""

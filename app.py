from __future__ import division, print_function
import sys
import os
import glob
import re
import numpy as np
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
import tensorflow
from keras.preprocessing import image
from keras.models import load_model
import cv2
import tensorflow as tf

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'covid19Xnormal.model'

# Load your trained model
model = tensorflow.keras.models.load_model(MODEL_PATH)
#model._make_predict_function()          # Necessary

print('Model loaded. Check http://127.0.0.1:5000/')


def model_predict(img_path, model):
    print('Hello world!', file=sys.stderr)
    img = image.load_img(img_path, target_size=(100, 100))
    categories  = ["covid19","normal"]

    def prepare(filepath):
        IMG_SIZE = 100
        img_array = cv2.imread(filepath,cv2.IMREAD_GRAYSCALE)
        new_array = cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
        return new_array.reshape(-1,IMG_SIZE,IMG_SIZE,1)

    model = tf.keras.models.load_model("covid19Xnormal.h5")
    prediction = model.predict([prepare(img)])
    print(prediction,file=stderr)
    print(str(categories[int(prediction[0][0])]),file=stderr)
    return prediction[0][0]


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['POST','GET'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        # Make prediction
        preds = model_predict(file_path, model)
        result = str(preds)
        #print(result,file = sys.stderr)
        return result
    else: return None


if __name__ == '__main__':
    app.run(debug=True)




import flask
from flask import Flask, render_template, request
import numpy as np
import keras
from keras.models import load_model
from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

@app.route('/')

@app.route('/index.html')
def index():
    return flask.render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    model = load_model('CROP_PRED_MODEL.h5')
    moisture = float(request.form['moisture'])
    nitrogen = float(request.form['nitrogen'])
    phosphorous = float(request.form['phosphorous'])
    potassium = float(request.form['potassium'])

    test_vector = [moisture,nitrogen,phosphorous,potassium]  # value for barley
    test_vector = np.asanyarray(test_vector)
    test_vector = np.reshape(test_vector, (1, 4))
    reverse_mapping = ['Barley', 'Corn-Field for silage', 'Corn-Field for stover', 'Millet', 'Potato', 'Sugarcane']
    reverse_mapping = np.asarray(reverse_mapping)
    a = model.predict_classes(test_vector)
    prediction = reverse_mapping[a]
    return render_template("predict.html",prediction=prediction)

if __name__ == '__main__':
    app.run()

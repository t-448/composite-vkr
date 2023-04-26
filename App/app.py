import flask
from flask import render_template
import tensorflow as tf
from tensorflow import keras
import numpy as np

app = flask.Flask(__name__, template_folder = 'templates')

def prediction(param1, param2, param3, param4, param5, param6, param7, param8, param9, param10, param11, param12):
    model = tf.keras.models.load_model('../models/model2/')
    prediction = model.predict([param1, param2, param3, param4, param5, param6, param7, param8, param9, param10, param11, param12])
    return prediction[0][0]

@app.route('/', methods=['POST', 'GET'])

def main():
    params_list = []
    result = ''    
    if flask.request.method == 'GET':
        return render_template('main.html')
    
    if flask.request.method == 'POST':
        for i in range(1,13,1):
            param = flask.request.form.get(f'param{i}')
            params_list.append(float(param))
            
        result = prediction(*params_list)
    return render_template('main.html', result=result)

if __name__ == '__main__':
    app.run()
import flask
from flask import Flask, request, render_template
import tensorflow as tf
from tensorflow import keras

app = flask.Flask(__name__, template_folder = 'templates')

def prediction(params):
    model = pickle.load('models/<model-name>')
    pred = model.predict([params])
    return pred

@app.route('/', methods=['POST', 'GET'])
def main():
    if flask.request.method == 'GET':
        return render_template('main.html')
    
    if flask.request.method == 'POST':
        model = tf.keras.models.load_model('../models/model2/')
        params = float(flask.request.form['plotnost', 'modul_up', 'kolvo_otverd', 'epoxs_group', 'temperature', 
                                              'poverh_plotnost', 'modul_up_rast', 'prochnost_rast', 'potreb_smol', 'ugol_nashivki', 
                                              'shag_nashivki', 'plotnost_nashivki'])   
        params = params.reshape(1,-1)
        result = model.predict([params])
    return render_template('templates/main.html', result=result)

if __name__ == '__main__':
    app.run()
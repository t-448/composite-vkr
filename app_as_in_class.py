import flask
from flask import render_template
import pickle
import tensorflow as tf

app = flask.Flask(__name__, template_folder = 'templates')

@app.route('/', methods=['POST', 'GET'])
def main():
    if flask.request.method == 'GET':
        return render_template('main.html')
    
    if flask.request.method == 'POST':
        with open('models/model_E.pkl', 'rb') as f: 
            loaded_model = pickle.load(f)
            
        params = float(flask.request.form['plotnost', 'modul_up', 'kolvo_otverd', 'epoxs_group', 'temperature', 
                                              'poverh_plotnost', 'modul_up_rast', 'prochnost_rast', 'potreb_smol', 'ugol_nashivki', 
                                              'shag_nashivki', 'plotnost_nashivki'])
        params = params.reshape(1,-1)
        result = loaded_model.predict([[params]])
    return render_template('main.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
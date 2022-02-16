import flask
from flask import request
from flask.templating import render_template
import numpy as np
import joblib
from joblib import dump, load
from io import BytesIO
import base64

app = flask.Flask(__name__, template_folder='templates')

model = joblib.load(open('model/model_ensemble.pkl', 'rb'))

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI IKG
    '''
    int_features = [int(x) for x in flask.request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0])

    return flask.render_template('main.html', prediction_text='The prediction is : {}  (0 for live, 1 for dead)'.format(output))

@app.route('/')
def main():
    return(flask.render_template('main.html'))
    
if __name__ == '__main__':
    app.run(debug=True)

    
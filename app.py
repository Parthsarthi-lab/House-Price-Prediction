import pickle
from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np
import pandas as pd

app = Flask(__name__)
model = pickle.load(open("model.pkl",'rb'))
scalar = pickle.load(open("scaling.pkl",'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data = request.json['data'] # Stores the data in JSON format
    print(data)
    print(np.array(list(data.values())).reshape(1,-1)) # The JSON format is stored as a dictionary. That's why we are using dict.values to get the data values
    new_data = scalar.transform(np.array(list(data.values())).reshape(1,-1))
    output = model.predict(new_data)
    print(output[0])
    return jsonify(output[0])

@app.route("/predict",methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    final_input = scalar.transform(np.array(data).reshape(1,-1))
    print(final_input)
    output= model.predict(final_input)[0]
    return render_template("home.html",prediction_text="The Predicted House Price is {}".format(output))



if __name__ == "__main__" :
    app.run(debug=True)

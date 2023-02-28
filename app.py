import numpy as np
import pandas as pd
from flask import *
from flask import Flask, request, render_template, jsonify, json
from flask_restx import reqparse
import pickle
from sklearn.metrics import accuracy_score, confusion_matrix
#import matplotlib
#import matplotlib.pyplot as plt

# Create the Flask App
app = Flask(__name__)
model = pickle.load(open('myModel.pkl', 'rb'))
cols = ['root_shell']
# EXAMPLE cols = ['Carat Weight', 'Cut', 'Color', 'Clarity', 'Polish', 'Symmetry', 'Report']

@app.route('/')
def main():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
       # root = request.form
        int_features = [float(x) for x in request.form.values()]
        final_features = np.array(int_features)
        print(final_features)
        #input_variables = pd.DataFrame([final]) 
        #columns=(['root_shell', 'num_access_files', 'num_failed_login'], dtype=float)
        #X_new = np.fromiter(input_variables, dtype=float, count = -1)  #test
        #parser = reqparse.RequestParser() #test
        #parser.add_argument('number')
        #args = parser.parse_args(strict=True)  # creates dict   TEst
    #   prediction = model.predict(args([X_new]))[0] #test
     #   prediction = model.predict(input_variables)   ### [0] original/ testing array size ORIGINAL

       # input_variables = request.get_json(force=True)
        #data_unseen = pd.DataFrame([input_variables])
        #prediction = model.predict(model, data=input_variables ) #test
        #prediction = prediction.Label[0]
       # return jsonify(prediction)

        #prediction = model.predict(args([X_new]))[0] #test
        prediction = model.predict(final_features)  ### [0] original/ testing array size ORIGINAL
     #   return jsonify(prediction)
        # Measure accuracy        
           # Return the components to the HTML template

        return render_template('index.html', result='The Intrusion Detection prediction is {}'.format(prediction))
 
    
        #cf_matrix = confusion_matrix(y_eval, prediction)
        #score = metrics.accuracy_score(y_eval, prediction)

if __name__ == "__main__":
    app.run(debug = True)

@app.route('/predict2')
def predict2():
    jsonfile = request.get_json()
     # Get the data from the POST request.
    data = request.get_json()    # Make prediction using model loaded from disk as per the data.
    data = pd.read_json(json.dumps(jsonfile),12)
    y_pred = model.predict(data)  # Take the first value of prediction
    output = y_pred.score[y_pred, data]   
    print("PREDICTION  ->> PREDICTION SCORE", output)
    return jsonify(output)

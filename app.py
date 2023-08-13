from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

filename = 'decisiontree.pkl'
classifier = pickle.load(open(filename,'rb'))
model = pickle.load(open('decisiontree.pkl','rb'))

app = Flask(__name__, template_folder= "templates") #template folder

@app.route('/',methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict_value():
    try:
        passenger = int(request.form['passenger_count'])
        distance = int(request.form['trip_distance'])
        time_taken = int(request.form['pick_hour'])
        payment1 = int(request.form['tot_mins_diff'])
        payment2 = int(request.form['cash1'])
        pick_hour = int(request.form['creditcard2'])

        input_features = [passenger, distance, time_taken, payment1, payment2, pick_hour]
        features_value = [np.array(input_features)]
        feature_name = ['passenger_count', 'trip_distance', 'pick_hour', 'tot_mins_diff','cash1',
                        'creditcard2']

        df = pd.DataFrame(features_value, columns=feature_name)
        output = model.predict(df)

        return render_template('index.html', prediction_text='Fare Prediction: {:.2f}'.format(output[0]))
    except Exception as e:
        return render_template('index.html', prediction_text='Error: {}'.format(e))

    
if __name__ == "__main__":
    app.run(debug=True)

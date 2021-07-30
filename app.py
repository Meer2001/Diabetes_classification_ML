from flask import Flask, render_template, request
import pickle
import numpy as np
import diabetesmodel

filename = 'diabetesmodel.pkl'
classifier = pickle.load(open(filename,'rb'))

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == "POST":
        preg = int(request.form['Pregnancies'])
        glucose = int(request.form['Glucose'])
        bp = int(request.form['BloodPressure'])
        skinthickness = int(request.form['SkinThickness'])
        insulin = int(request.form['Insulin'])
        bmi = float(request.form['BMI'])
        diabetespf = float(request.form['DiabetesPedigreeFunction'])
        age = int(request.form['Age'])
        
        data = np.array([[preg, glucose, bp, skinthickness, insulin, bmi, diabetespf, age]])
        result = classifier.predict(data)
        
        return render_template('result.html', prediction=result)
    
if __name__ == "__main__":
    app.run(debug=True)



import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import json, requests
import pandas as pd
import numpy as np
import joblib
import base64
import io
from flask import Flask, render_template, url_for, redirect, request, abort
import seaborn as sns
plt.style.use('fivethirtyeight')

app = Flask(__name__, static_url_path='')

df = pd.read_csv(
    'diabetes.csv',
    names = ['P','G','BP','ST','I','BMI','DPF','A','O'],
    header = 0
)

@app.route('/')
@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/prediction')
def prediction():
    return render_template('prediction.html')

@app.route('/result', methods = ['GET', 'POST'])
def result():
    if request.method == 'POST':
        P = int(request.form['P'])
        G = int(request.form['G'])
        BP = int(request.form['BP'])
        ST = int(request.form['ST'])
        I = int(request.form['I'])
        BMI = float(request.form['BMI'])
        DPF = float(request.form['DPF'])
        A = int(request.form['A'])

        if float(DPF) >=2:
            return render_template('error.html')
        elif float(DPF) < 0:
            return render_template('error.html') 
            
        prediction = model.predict([[
            P, G, BP, ST, I, BMI, DPF, A
            ]])[0]

        if prediction == 0:
            preds = "You're Healthy"
        else:
            preds = "Positive Diabetes! You need to take a medication!"
        
        predictResult = {'P': P , 'G': G , 'BP': BP, 'ST': ST, 'I': I, 'BMI': BMI, 'DPF': DPF, 'A': A, 'preds': preds}

        return render_template('result.html', result = predictResult)
    else:
        return render_template('error.html')

@app.route('/medication')
def medication():
    return render_template('medication.html')

@app.route('/medication2')
def medication2():
    return render_template('medication2.html')
    
@app.route('/graph')
def graph():
    return render_template('graph.html')

@app.errorhandler(404)
def page_not_found(error):
	return render_template('error.html')

if __name__ == "__main__":
    model = joblib.load('modelDiabetes')
    app.run(debug=True)

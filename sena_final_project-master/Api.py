from flask import Flask, request, jsonify, render_template
import os
import numpy as np
import utils

app = Flask(__name__)


@app.route('/')
def hello_world():
    return render_template('index.html')


@app.route('/classify_sentiments', methods=['GET', 'POST'])
def tweeter_senti():
    if request.method == 'POST':
        senti = request.form['sentence']
        pred = utils.making_prediction(senti)
        group = np.argmax(pred)
        print(group)
        output = ""
        if group == 0:
            output = "negatif"
        elif group == 1:
            output = "notr"
        elif group == 2:
            output = "positif"

        return render_template("index.html", message=f' {output} '.upper())


if __name__ == '__main__':
    app.run()

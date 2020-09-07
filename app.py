# USAGE
# Start the server:
# 	python app.py

# import the necessary packages
import numpy as np
import pandas as pd
import io, os
from flask import Flask, jsonify, render_template, request, flash, redirect, url_for
import requests
from joblib import load
from xgboost import XGBClassifier

# initialize our Flask application
app = Flask(__name__, static_url_path='/static')
app.secret_key = os.urandom(24)

model_binary = None
model_multi = None

def load_model_binary():
	global model_binary
	# load model from file
	model_binary = load("./models/scadi-binary.model")
	print("Successfully loaded model: scadi-binary.model")

def load_model_multi():
	global model_multi
	# load model from file
	model_multi = load("./models/scadi-multi.model")
	print("Successfully loaded model: scadi-multi.model")

@app.route("/")
def index():
	ip = request.environ.get('HTTP_X_REAL_IP', request.remote_addr)
	app.logger.info('%s just accessed your webapp', ip)

	return render_template('index.html')

@app.route("/binary")
def binary():
	ip = request.environ.get('HTTP_X_REAL_IP', request.remote_addr)
	app.logger.info('%s just accessed your webapp', ip)

	return render_template('binary.html')

@app.route("/multi")
def multi():
	ip = request.environ.get('HTTP_X_REAL_IP', request.remote_addr)
	app.logger.info('%s just accessed your webapp', ip)

	return render_template('multi.html')

@app.route("/binary/result", methods=["POST"])
def binary_result():
	ip = request.environ.get('HTTP_X_REAL_IP', request.remote_addr)
	app.logger.info('%s just post the data', ip)
	
	formdata = request.form.to_dict(flat=False)
	dtInput = pd.DataFrame.from_dict(formdata,orient='columns').astype(int)
	classtype = 'binary'
	
	predictions = model_binary.predict(dtInput)
	if predictions == 1:
		status = 'Having self-care problem (1)'
	else:
		status = 'Not having self-care problem (0)'
	
	return render_template('result.html', classtype=classtype, status=status)


@app.route("/multi/result", methods=["POST"])
def multi_result():
	ip = request.environ.get('HTTP_X_REAL_IP', request.remote_addr)
	app.logger.info('%s just post the data', ip)

	formdata = request.form.to_dict(flat=False)
	dtInput = pd.DataFrame.from_dict(formdata,orient='columns').astype(int)
	classtype = 'multi'
	
	predictions = model_multi.predict(dtInput)
	if predictions == 1:
		status = 'Having self-care problem (1)'
	else:
		status = 'Not having self-care problem (0)'
	
	return render_template('result.html', classtype=classtype, status=status)

# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
	print(("* Loading the models and Flask starting server..."
		"please wait until server has fully started"))
	load_model_binary()
	load_model_multi()
	app.run(host='localhost',port=2097,debug=True)
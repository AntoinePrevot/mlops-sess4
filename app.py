from flask import Flask
import joblib
import pandas as pd

path_to_model="model.v0.pickle"
model = joblib.load(path_to_model)

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"


@app.route("/predict/<name>")
def predict(name):
    pred = model.predict(encoder(pd.Series([name.upper()])))
    if pred == 0:
        res = "fille"
    else:
        res = "garcon"
    return {"hello": "hello", name: res}


def encoder(names):

  alphabet = "abcdefghijklmnopqrstuvwxyz".upper()
  features = pd.DataFrame()
  for letter in alphabet:
    features[letter] = names.str.count(letter).astype(int)

  return features

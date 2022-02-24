from flask import Flask
import joblib
import pandas as pd

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"


@app.route("/predict/<name>")
def predict(name, path_to_model="model.v0.pickle"):
    model = joblib.load(path_to_model)
    return model.predict(encoder(pd.Series[name]))


def encoder(names):

  alphabet = "abcdefghijklmnopqrstuvwxyz".upper()
  features = pd.DataFrame()
  for letter in alphabet:
    features[letter] = names.str.count(letter).astype(int)

  return features

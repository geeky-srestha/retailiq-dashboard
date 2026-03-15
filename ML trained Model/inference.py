import pickle
import json
import numpy as np

def model_fn(model_dir):
    model = pickle.load(open(model_dir + "/sentiment_model.pkl", "rb"))
    vectorizer = pickle.load(open(model_dir + "/vectorizer.pkl", "rb"))
    return (model, vectorizer)

def predict_fn(input_data, model):
    model, vectorizer = model
    text = [input_data["review"]]
    X = vectorizer.transform(text)
    prediction = model.predict(X)
    return prediction.tolist()
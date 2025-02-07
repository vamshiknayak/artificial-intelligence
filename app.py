from flask import Flask, request, jsonify
import pickle
import numpy as np

# Load the saved model
model = pickle.load(open("taxifare_model.pkl", "rb"))

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = np.array([data['latitude_of_pickup'], data['longitude_of_pickup'], 
                         data['latitude_of_dropoff'], data['longitude_of_dropoff'], data['Distance']]).reshape(1, -1)
    prediction = model.predict(features)
    
    return jsonify({"Predicted Fare": prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)

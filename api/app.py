from flask import Flask, request, jsonify
from flask_cors import CORS 
import pickle
import numpy as np

app = Flask(__name__)
CORS(app)

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict_price():
    try:
        data = request.get_json()
        house_size = data.get("house_size")
        num_rooms = data.get("num_rooms")

        if house_size is None or num_rooms is None:
            return jsonify({"error": "Missing required parameters: house_size and num_rooms"}), 400

        prediction = float(model.predict(np.array([[house_size, num_rooms]]))[0])
        return jsonify({"predicted_price": round(prediction, 2)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

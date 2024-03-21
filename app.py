from flask import Flask, request, jsonify
import joblib

# Load model
model = joblib.load("model.pkl")

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
  # Get data from request
  data = request.get_json()
  
  # Convert data to a pandas DataFrame
  df = pd.DataFrame(data)
  
  # Make prediction
  prediction = model.predict(df)[0]
  
  # Return response
  return jsonify({"predicted_price": prediction})

if __name__ == "__main__":
  app.run(debug=True)
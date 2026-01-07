from flask import Flask, request, jsonify
from api.inference  import predict_job_fraud
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/health', methods=["GET"])
def health():
    return jsonify({"status": "ok"})

@app.route('/predict', methods=["POST"])
def predict():
    data = request.json

    result = predict_job_fraud(title = data.get("title", ""),
                               company_profile=data.get("company_profile", ""),
                               description=data.get("description", ""),
                               requirements=data.get("requirements", ""),
                               benefits=data.get("benefits", ""))

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
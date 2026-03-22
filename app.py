from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the saved pipeline dictionary
with open('model.pkl', 'rb') as f:
    pipeline = pickle.load(f)

scaler        = pipeline['scaler']
kmeans        = pipeline['kmeans']
CLUSTER_LABELS = pipeline['cluster_labels']
FEATURE_NAMES  = pipeline['feature_names']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    # Extract features in exact order from training
    features = np.array([[
        data['child_mort'],
        data['exports'],
        data['health'],
        data['imports'],
        data['income'],
        data['inflation'],
        data['life_expec'],
        data['total_fer'],
        data['gdpp']
    ]])

    # Scale then predict cluster
    features_scaled = scaler.transform(features)
    cluster_id      = int(kmeans.predict(features_scaled)[0])
    cluster_label   = CLUSTER_LABELS[cluster_id]

    return jsonify({
        'cluster':    cluster_id,
        'category':   cluster_label,
        'features':   FEATURE_NAMES
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
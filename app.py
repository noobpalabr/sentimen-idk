import os
from flask import Flask, request, render_template, jsonify
import pickle

app = Flask(__name__)

# Load model
with open('svm_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Load vectorizer
with open('vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    texts = request.form['texts'].split('\n')
    
    # Preprocess text
    transformed_texts = vectorizer.transform(texts)
    
    # Predict
    predictions = model.predict(transformed_texts)
    
    # Format predictions
    results = predictions.tolist()
    
    return render_template('index.html', predictions=results)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)

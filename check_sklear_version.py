import pickle
from sklearn import __version__ as sklearn_version

# Load model
with open('svm_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Print the scikit-learn version used to save the model
print(f"Current scikit-learn version: {sklearn_version}")

try:
    model_version = model.__getstate__()['_sklearn_version']
    print(f"Model saved with scikit-learn version: {model_version}")
except KeyError:
    print("Model version not available. It might have been saved with an older version of scikit-learn.")
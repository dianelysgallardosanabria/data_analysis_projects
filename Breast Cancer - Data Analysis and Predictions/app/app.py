import gradio as gr
import numpy as np
import joblib

# Load the Logistic Regression model
model = joblib.load('app/breast_cancer_model.joblib')

# Load the scaler from the file
scaler = joblib.load('app/scaler.pkl')

# Define the names of the features
feature_names = [
    'radius_mean', 'texture_mean', 'perimeter_mean',
       'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
       'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
       'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
       'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
       'fractal_dimension_se', 'radius_worst', 'texture_worst',
       'perimeter_worst', 'area_worst', 'smoothness_worst',
       'compactness_worst', 'concavity_worst', 'concave points_worst',
       'symmetry_worst', 'fractal_dimension_worst'
]
feature_ranges = [
   (0, 30), (0, 40), (0, 200), (0, 2600), (0, 0.2),
    (0, 0.4), (0, 0.5), (0, 0.3), (0, 0.4), (0, 0.15),
    (0, 3), (0, 4), (0, 25), (0, 600), (0, 0.04),
    (0, 0.2), (0, 0.2), (0, 0.1), (0, 0.1), (0, 0.03),
    (0, 40), (0, 50), (0, 300), (0, 4500), (0, 0.3),
    (0, 1.2), (0, 1.3), (0, 0.4), (0, 0.7), (0, 0.3)
]

def predict(*features):
    # Convert features into a numpy array and reshape it for prediction
    features = np.array(features).reshape(1, -1)
    
    # Scale the features using the loaded scaler
    scaled_features = scaler.transform(features)
    
    # Make a prediction and retrieve the prediction and probability
    prediction = model.predict(scaled_features)[0]
    probability = model.predict_proba(scaled_features)[0].max()
    
    # Define advice based on the prediction
    advice = ""
    if prediction == 1:  # Assuming 1 represents 'Malignant'
        advice = "We advise consulting a healthcare professional immediately."
    else:
        advice = "Results are generally non-alarming. However, regular check-ups are recommended."
    
    # Separate outputs for prediction, probability, and advice
    prediction_output = 'Malignant' if prediction == 1 else 'Benign'
    probability_output = f"{probability:.2%}"
    advice_output = advice
    
    return prediction_output, probability_output, advice_output

# Set up sliders for each feature with appropriate min/max values
inputs = [gr.Slider(minimum=min_val, maximum=max_val, label=fname, step=(max_val-min_val)/100) 
          for fname, (min_val, max_val) in zip(feature_names, feature_ranges)]
outputs = [
    gr.Label(label="Prediction"),
    gr.Label(label="Probability"),
    gr.Textbox(label="Advice")
]

interface = gr.Interface(
    fn=predict, 
    inputs=inputs, 
    outputs=outputs, 
    title="Breast Cancer Prediction",
    description="Slide to set feature values:",
    theme = gr.themes.Soft()
).launch()

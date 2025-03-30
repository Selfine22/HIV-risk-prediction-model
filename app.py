from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
with open("model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# Load label encoders used for categorical features
with open("label_encoders.pkl", "rb") as encoders_file:
    label_encoders = pickle.load(encoders_file)

# Define feature order (Make sure these match index.html exactly)
FEATURES = [
    "age", "marital staus", "std", "educational background", 
    "hiv test in past year", "aids education", 
    "places of seeking sex partners", "sexual orientation", "drug- taking"
]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Retrieve form data (ensure names match index.html)
        input_data = {}
        for feature in FEATURES:
            value = request.form.get(feature, "").strip()
            if not value:
                return jsonify({"error": f"Missing value for {feature}."})

            # Convert numeric fields properly
            if feature == "age":  # Ensure numerical fields are converted
                try:
                    input_data[feature] = float(value)
                except ValueError:
                    return jsonify({"error": f"Invalid input for {feature}. Must be a number."})
            else:
                input_data[feature] = value

        # Convert categorical variables using label encoders
        for feature, encoder in label_encoders.items():
            if feature in input_data:
                classes = list(encoder.classes_)  # Original categories
                label_mapping = {str(cls): idx for idx, cls in enumerate(classes)}  # Mapping to numbers

                if input_data[feature] in label_mapping:
                    input_data[feature] = label_mapping[input_data[feature]]
                else:
                    return jsonify({"error": f"Invalid input for {feature}: {input_data[feature]}. Expected one of {classes}."})

        # Convert to NumPy array
        input_array = np.array(list(input_data.values())).reshape(1, -1)

        ## Make prediction
        prediction = model.predict(input_array)
        result = "High risk" if prediction[0] == 1 else "Low risk"

        return jsonify({"Prediction": result})
    
    except Exception as e:
        return jsonify({"error": str(e)})
if __name__ == '__main__':
    app.run(debug=True)
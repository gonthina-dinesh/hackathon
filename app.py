from flask import Flask, request, render_template
import pickle
import pandas as pd

# Load the model, scaler, and encoder
with open("model_heart.pkl", "rb") as file:
    model_heart = pickle.load(file)

with open("scaler_heart.pkl", "rb") as file:
    scaler_heart = pickle.load(file)

with open("enc_heart.pkl", "rb") as file:
    enc_heart = pickle.load(file)

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Collect data from the form
        data = {
            "Age": float(request.form["Age"]),
            "Sex": request.form["Sex"],
            "ChestPainType": request.form["ChestPainType"],
            "RestingBP": float(request.form["RestingBP"]),
            "Cholesterol": float(request.form["Cholesterol"]),
            "FastingBS": float(request.form["FastingBS"]),
            "RestingECG": request.form["RestingECG"],
            "MaxHR": float(request.form["MaxHR"]),
            "ExerciseAngina": request.form["ExerciseAngina"],
            "Oldpeak": float(request.form["Oldpeak"]),
            "ST_Slope": request.form["ST_Slope"],
        }

        # Convert data into DataFrame
        input_df = pd.DataFrame([data])

        # Preprocess inputs
        numeric_cols = ["Age", "RestingBP", "Cholesterol", "FastingBS", "MaxHR", "Oldpeak"]
        categorical_cols = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]

        # Scale numeric columns
        input_df[numeric_cols] = scaler_heart.transform(input_df[numeric_cols])

        # Encode categorical columns
        encoded_categorical = enc_heart.transform(input_df[categorical_cols])
        encoded_categorical = pd.DataFrame(
            encoded_categorical, columns=enc_heart.get_feature_names_out(categorical_cols)
        )

        # Combine numeric and encoded categorical features
        input_preprocessed = pd.concat([input_df[numeric_cols], encoded_categorical], axis=1)

        # Make prediction
        prediction = model_heart.predict(input_preprocessed)
        output = "at risk of a heart attack" if prediction[0] == 1 else "not at risk of a heart attack"

        # Return the result
        return render_template("index.html", prediction_text=f"The person is {output}.")
    
    except Exception as e:
        return render_template("index.html", prediction_text=f"An error occurred: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)

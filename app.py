from flask import Flask, render_template, request
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


app = Flask(__name__)


# ==============================
# LOAD DATASET
# ==============================

data = pd.read_csv(r"C:\Users\USER\Downloads\WA_Fn-UseC_-Telco-Customer-Churn.csv")


# ==============================
# DATA PREPROCESSING
# ==============================

# Convert TotalCharges to numeric
data["TotalCharges"] = pd.to_numeric(
    data["TotalCharges"],
    errors="coerce"
)

# Remove missing values
df = data.dropna()

# Remove customer ID
df = df.drop("customerID", axis=1)

# Convert categorical columns into numerical columns
df = pd.get_dummies(
    df,
    columns=df.select_dtypes(include="object").columns,
    drop_first=True
)


# ==============================
# X AND Y
# ==============================

X = df.drop("Churn_Yes", axis=1)
y = df["Churn_Yes"]


# ==============================
# TRAIN TEST SPLIT
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ==============================
# FEATURE SCALING
# ==============================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ==============================
# TRAIN MODEL
# ==============================

model = LogisticRegression(max_iter=1000)

model.fit(
    X_train_scaled,
    y_train
)


# ==============================
# MODEL ACCURACY
# ==============================

y_pred = model.predict(X_test_scaled)

accuracy = accuracy_score(
    y_test,
    y_pred
)


# ==============================
# HOME PAGE
# ==============================

@app.route("/")
def home():

    return render_template(
        "index.html",
        accuracy=round(accuracy * 100, 2)
    )


# ==============================
# PREDICTION
# ==============================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        # Get values from form

        gender = request.form["gender"]

        senior_citizen = int(
            request.form["senior_citizen"]
        )

        partner = request.form["partner"]

        dependents = request.form["dependents"]

        tenure = int(
            request.form["tenure"]
        )

        monthly_charges = float(
            request.form["monthly_charges"]
        )

        total_charges = float(
            request.form["total_charges"]
        )

        contract = request.form["contract"]

        internet_service = request.form[
            "internet_service"
        ]


        # Create empty dataframe
        input_data = pd.DataFrame(
            columns=X.columns
        )

        input_data.loc[0] = 0


        # Numerical values

        if "SeniorCitizen" in input_data.columns:
            input_data["SeniorCitizen"] = senior_citizen

        if "tenure" in input_data.columns:
            input_data["tenure"] = tenure

        if "MonthlyCharges" in input_data.columns:
            input_data["MonthlyCharges"] = monthly_charges

        if "TotalCharges" in input_data.columns:
            input_data["TotalCharges"] = total_charges


        # Categorical values

        if gender == "Male":
            if "gender_Male" in input_data.columns:
                input_data["gender_Male"] = 1


        if partner == "Yes":
            if "Partner_Yes" in input_data.columns:
                input_data["Partner_Yes"] = 1


        if dependents == "Yes":
            if "Dependents_Yes" in input_data.columns:
                input_data["Dependents_Yes"] = 1


        # Contract

        if contract == "One year":
            if "Contract_One year" in input_data.columns:
                input_data["Contract_One year"] = 1

        elif contract == "Two year":
            if "Contract_Two year" in input_data.columns:
                input_data["Contract_Two year"] = 1


        # Internet Service

        if internet_service == "Fiber optic":
            if "InternetService_Fiber optic" in input_data.columns:
                input_data[
                    "InternetService_Fiber optic"
                ] = 1

        elif internet_service == "No":
            if "InternetService_No" in input_data.columns:
                input_data[
                    "InternetService_No"
                ] = 1


        # Scale input
        input_scaled = scaler.transform(input_data)


        # Prediction
        prediction = model.predict(
            input_scaled
        )[0]


        # Probability
        probability = model.predict_proba(
            input_scaled
        )[0][1] * 100


        if prediction == 1:

            result = "Customer is likely to CHURN"

        else:

            result = "Customer is likely to STAY"


        return render_template(
            "index.html",
            prediction=result,
            probability=round(probability, 2),
            accuracy=round(accuracy * 100, 2)
        )


    except Exception as e:

        return render_template(
            "index.html",
            prediction="Error: " + str(e),
            accuracy=round(accuracy * 100, 2)
        )


if __name__ == "__main__":
    app.run(debug=True)
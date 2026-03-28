import numpy as np
import pandas as pd
import joblib

from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Load model
model = joblib.load("loan_default_model.pkl")


@app.route("/")
def splash():
    return render_template("splash.html")


@app.route("/dashboard")
def dashboard():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # ===== INPUT =====
        age = int(request.form["Age"])
        education = request.form["Education"]
        marital_status = request.form["MaritalStatus"]
        dependents = request.form["HasDependents"]

        income = float(request.form["Income"])
        credit_score = int(request.form["CreditScore"])
        dti_ratio = float(request.form["DTIRatio"])
        num_credit_lines = int(request.form["NumberCreditLines"])  # 👈 FIXED NAME
        has_mortgage = request.form["HasMortgage"]

        employment_type = request.form["EmploymentType"]
        months_employed = int(request.form["MonthsEmployed"])

        loan_amount = float(request.form["LoanAmount"])
        interest_rate = float(request.form["InterestRate"])
        loan_term = int(request.form["LoanTerm"])
        loan_purpose = request.form["LoanPurpose"]
        has_cosigner = request.form["HasCoSigner"]

        # ===== FEATURE ENGINEERING =====
        loan_income_ratio = loan_amount / income if income != 0 else 0
        employment_ratio = months_employed / age if age != 0 else 0
        credit_load = dti_ratio * num_credit_lines

        data = pd.DataFrame([{
            "Age": age,
            "Education": education,
            "MaritalStatus": marital_status,
            "HasDependents": dependents,
            "Income": income,
            "CreditScore": credit_score,
            "DTIRatio": dti_ratio,
            "NumCreditLines": num_credit_lines,
            "HasMortgage": has_mortgage,
            "EmploymentType": employment_type,
            "MonthsEmployed": months_employed,
            "LoanAmount": loan_amount,
            "InterestRate": interest_rate,
            "LoanTerm": loan_term,
            "LoanPurpose": loan_purpose,
            "HasCoSigner": has_cosigner,
            "LoanIncomeRatio": loan_income_ratio,
            "EmploymentRatio": employment_ratio,
            "CreditLoad": credit_load
        }])

        # ===== PREDICTION =====
        prediction = model.predict(data)[0]
        probability = model.predict_proba(data)[0][1]

        if prediction == 1:
            result = "High Risk of Default"
        else:
            result = "Low Risk (Loan Likely Safe)"

        # ✅ RETURN JSON (IMPORTANT)
        return render_template(
            "result.html",
            prediction=result,
            probability=round(probability, 3)
        )

    except Exception as e:
        return jsonify({
            "prediction": "Error",
            "message": str(e)
        })


if __name__ == "__main__":
    app.run(debug=True)
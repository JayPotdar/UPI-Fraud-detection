import joblib
import pandas as pd

#load saved artifacts
preprocessor = joblib.load("models/preprocessor.pkl")
print("Features expected by preprocessor:")
print(preprocessor.feature_names_in_)
model = joblib.load("models/logistic_regression_model1.pkl")
threshold = joblib.load("models/final_threshold1.pkl")


def predict_fraud(transaction: dict):
    #convert one transaction into data frame
    input_df = pd.DataFrame([transaction])
    
    print("Input columns:")
    print(input_df.columns.tolist())
    
    prepared_data = preprocessor.transform(input_df)
    
    fraud_score = model.predict_proba(prepared_data)[:,1][0]
    
    prediction = int(fraud_score >= threshold)
    
    return {
        "fraud_score": float(fraud_score),
        "threshold": float(threshold),
        "prediction": prediction,
        "prediction_label": "Fraud" if prediction == 1 else "Genuine"
    }
    
if __name__ == "__main__":

    sample_transaction = {
        "transaction type": "P2P",
        "merchant_category": "Food",
        "amount (INR)": 244,
        "sender_age_group": "26-35",
        "receiver_age_group": "26-35",
        "sender_state": "Maharashtra",
        "sender_bank": "ICICI",
        "receiver_bank": "HDFC",
        "device_type": "Android",
        "network_type": "4G",
        "transaction_hour": 15,
        "day_of_week": "Saturday",
        "is_weekend": 1,
        "night_transaction": 0,
        "same_bank": 0,
        "same_age_group": 1,
        "high_value_transaction": 0,
        "peak_hour": 0
    }

    result = predict_fraud(sample_transaction)

    print(result)
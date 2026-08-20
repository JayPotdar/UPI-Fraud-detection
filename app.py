from fastapi import FastAPI
from pydantic import BaseModel, Field
from inference import predict_fraud
from typing import Literal


app = FastAPI(
    title="UPI Fraud Detection API",
    description=(
        "Machine learning API for generating a fraud-risk score "
        "for UPI transactions."
    ),
    version="1.0.0"
)

class PredictionResponse(BaseModel):
    fraud_score: float
    threshold: float
    prediction: int
    prediction_label: Literal["Fraud", "Genuine"]

class Transaction(BaseModel):
    transaction_type: Literal[
        "P2P",
        "P2M",
        "Bill Payment",
        "Recharge"
    ]

    merchant_category: str

    amount_INR: float = Field(
        gt=0,
        description="Transaction amount must be greater than 0"
    )

    sender_age_group: Literal[
        "18-25",
        "26-35",
        "36-45",
        "46-55",
        "56+"
    ]

    receiver_age_group: Literal[
        "18-25",
        "26-35",
        "36-45",
        "46-55",
        "56+"
    ]

    sender_state: str

    sender_bank: str
    receiver_bank: str

    device_type: Literal[
        "Android",
        "iOS",
        "Web"
    ]

    network_type: Literal[
        "3G",
        "4G",
        "5G",
        "WiFi"
    ]

    transaction_hour: int = Field(
        ge=0,
        le=23
    )

    day_of_week: Literal[
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday"
    ]
    
    
    
@app.get("/")
def home():
    return {
        "message":"UPI Fraud Detection API is running"
    }
    
@app.post("/predict",
    response_model=PredictionResponse,
    summary="Predict UPI transaction fraud"
)
def predict(transaction: Transaction):
    
    # Feature Engineering
   
    # Weekend
    is_weekend = int(
        transaction.day_of_week in ["Saturday", "Sunday"]
    )

    # Night transaction: 12 AM - 5 AM
    night_transaction = int(
        0 <= transaction.transaction_hour <= 5
    )

    # Same bank
    same_bank = int(
        transaction.sender_bank == transaction.receiver_bank
    )

    # Same age group
    same_age_group = int(
        transaction.sender_age_group == transaction.receiver_age_group
    )

    # High-value transaction
    high_value_transaction = int(
        transaction.amount_INR > 4687.05
    )

    # Peak hour
    peak_hour = int(
        (10 <= transaction.transaction_hour <= 12)
        or
        (16 <= transaction.transaction_hour <= 20)
    )

    # -----------------------------
    # Build model input
    
        
    transaction_dict = {
    "transaction type": transaction.transaction_type,
    "merchant_category": transaction.merchant_category,
    "amount (INR)": transaction.amount_INR,
    "sender_age_group": transaction.sender_age_group,
    "receiver_age_group": transaction.receiver_age_group,
    "sender_state": transaction.sender_state,
    "sender_bank": transaction.sender_bank,
    "receiver_bank": transaction.receiver_bank,
    "device_type": transaction.device_type,
    "network_type": transaction.network_type,
    "transaction_hour": transaction.transaction_hour,
    "day_of_week": transaction.day_of_week,

    
    "is_weekend": is_weekend,
    "night_transaction": night_transaction,
    "same_bank": same_bank,
    "same_age_group": same_age_group,
    "high_value_transaction": high_value_transaction,
    "peak_hour": peak_hour
    }
    

    result = predict_fraud(transaction_dict)

    return result
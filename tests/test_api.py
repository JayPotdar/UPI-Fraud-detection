from fastapi.testclient import TestClient
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app import app

client = TestClient(app)


def test_home():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["message"] == "UPI Fraud Detection API is running"


def test_predict_valid_transaction():

    payload = {
        "transaction_type": "P2P",
        "merchant_category": "Food",
        "amount_INR": 244,
        "sender_age_group": "26-35",
        "receiver_age_group": "26-35",
        "sender_state": "Maharashtra",
        "sender_bank": "ICICI",
        "receiver_bank": "HDFC",
        "device_type": "Android",
        "network_type": "4G",
        "transaction_hour": 15,
        "day_of_week": "Saturday"
    }

    response = client.post(
        "/predict",
        json=payload
    )

    assert response.status_code == 200

    data = response.json()

    assert "fraud_score" in data
    assert "threshold" in data
    assert "prediction" in data
    assert "prediction_label" in data

    assert data["prediction"] in [0, 1]
    assert data["prediction_label"] in ["Fraud", "Genuine"]


def test_invalid_transaction_hour():

    payload = {
        "transaction_type": "P2P",
        "merchant_category": "Food",
        "amount_INR": 244,
        "sender_age_group": "26-35",
        "receiver_age_group": "26-35",
        "sender_state": "Maharashtra",
        "sender_bank": "ICICI",
        "receiver_bank": "HDFC",
        "device_type": "Android",
        "network_type": "4G",
        "transaction_hour": 30,
        "day_of_week": "Saturday"
    }

    response = client.post(
        "/predict",
        json=payload
    )

    assert response.status_code == 422
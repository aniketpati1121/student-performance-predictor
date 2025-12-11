import os
import pandas as pd
import logging
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Paths
PROCESSED_DATA_DIR = os.path.join("artifacts", "processed_data")
MODEL_DIR = os.path.join("artifacts", "model")
LOG_DIR = "logs"

MODEL_PATH = os.path.join(MODEL_DIR, "student_model.pkl")
SCALER_PATH = os.path.join("artifacts", "scaler.pkl")

# Create directories
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Logging config
logging.basicConfig(
    filename=os.path.join(LOG_DIR, "training.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def train_model():
    try:
        logging.info("Starting model training...")

        # Load processed data
        X_train = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, "X_train.csv"))
        X_test = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, "X_test.csv"))
        y_train = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, "Y_train.csv"))
        y_test = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, "Y_test.csv"))

        logging.info(f"Training data loaded: X_train: {X_train.shape}")
        logging.info(f"Testing data loaded: X_test: {X_test.shape}")

        # Scale the data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train model
        model = LinearRegression()
        model.fit(X_train_scaled, y_train)

        # Predictions
        y_pred = model.predict(X_test_scaled)

        # Evaluation
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        logging.info(f"Model Evaluation - MSE: {mse}")
        logging.info(f"Model Evaluation - R2 Score: {r2}")

        # Save model + scaler
        joblib.dump(model, MODEL_PATH)
        joblib.dump(scaler, SCALER_PATH)

        logging.info(f"Model saved at {MODEL_PATH}")
        logging.info(f"Scaler saved at {SCALER_PATH}")

        print("Model training completed successfully!")
        print(f"MSE: {mse}")
        print(f"R2 Score: {r2}")

    except Exception as e:
        logging.error(f"Error in training: {e}")
        raise e

if __name__ == "__main__":
    train_model()

import os
import pandas as pd
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

# PATHS 

RAW_DATA_PATH = os.path.join("artifacts", "raw_data", "raw_student_data.csv")
PROCESSED_DATA_DIR = os.path.join("artifacts", "processed_data")
LOG_DIR = "logs"
SCALER_PATH = os.path.join("artifacts", "scaler.pkl")

# CREATE DIRS 

os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# LOGGING CONFIG 

logging.basicConfig(
    filename=os.path.join(LOG_DIR, "preprocessing.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# PREPROCESS FUNCTION 

def preprocess_data():
    try:
        logging.info("Starting preprocessing...")

        # Load data
        df = pd.read_csv(RAW_DATA_PATH)
        logging.info(f"Data loaded for preprocessing: {df.shape}")

        # Fill missing values
        df.ffill(inplace=True)

        # Encode categorical columns
        le = LabelEncoder()
        if 'gender' in df.columns:
            df['gender'] = le.fit_transform(df['gender'])

        if 'parent_education' in df.columns:
            df['parent_education'] = le.fit_transform(df['parent_education'])

        # Drop useless columns
        columns_to_drop = []
        if 'name' in df.columns:
            columns_to_drop.append('name')
        if 'student_id' in df.columns:
            columns_to_drop.append('student_id')

        df.drop(columns=columns_to_drop, inplace=True)

        # Features and target
        X = df.drop(columns=["previous_score"])
        y = df["previous_score"]

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Feature scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Save processed data
        pd.DataFrame(X_train_scaled).to_csv(
            os.path.join(PROCESSED_DATA_DIR, "X_train.csv"), index=False
        )
        pd.DataFrame(X_test_scaled).to_csv(
            os.path.join(PROCESSED_DATA_DIR, "X_test.csv"), index=False
        )
        y_train.to_csv(
            os.path.join(PROCESSED_DATA_DIR, "y_train.csv"), index=False
        )
        y_test.to_csv(
            os.path.join(PROCESSED_DATA_DIR, "y_test.csv"), index=False
        )

        # Save scaler
        joblib.dump(scaler, SCALER_PATH)

        logging.info("Preprocessing completed successfully")
        print("Preprocessing completed successfully!")

    except Exception as e:
        logging.error(f"Error in preprocessing: {e}")
        raise e


# RUN SCRIPT 

if __name__ == "__main__":
    preprocess_data()

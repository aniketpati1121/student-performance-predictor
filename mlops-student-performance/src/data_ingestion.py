import os 
import pandas as pd 
import logging 
from datetime import datetime

#configurations
DATA_DIR = "data" 
RAW_DATA_DIR = os.path.join("artifacts", "raw_data")
LOG_DIR = "logs" 
DATA_FILE = os.path.join(DATA_DIR, "student_data.csv")

# Create necessary directories if not exist 
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# LOGGING CONFIGURATION
log_file = os.path.join(LOG_DIR, f"data_ingestion_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# DATA INGESTION FUNCTION 

def ingest_data(input_path: str, output_dir: str):
    try: 
        logging.info("Starting data ingestion...")

        # LOAD DATA 
        df = pd.read_csv(input_path)
        logging.info(f"Data loaded succesfully with shape: {df.shape}")

        # CHECK FOR MISSING VALUES 
        missing = df.isnull().sum()
        logging.info(f"Total missing values in dataset: {missing}")

        # SAVE RAW DATA AS ARTIFACT
        output_file = os.path.join(output_dir, "raw_student_data.csv")
        df.to_csv(output_file, index=False)
        logging.info(f"RAW data saved to {output_file}")

        logging.info("DATA ingestion completed succesfully.")
        return df 
    
    except Exception as e:
        logging.error(f"Error in data ingestion: {e}")
        raise e
    

if __name__ == "__main__":
    df = ingest_data(DATA_FILE, RAW_DATA_DIR)
    print(f"Data Ingestion Conpleted! Data Shape: {df.shape}")


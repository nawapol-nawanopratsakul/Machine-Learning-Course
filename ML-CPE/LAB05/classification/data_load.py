import pandas as pd
import os

def load_dataset(file_path="train.csv", sample_size=5000, random_state=42):

    print("Loading dataset...")
    

    current_dir = os.path.dirname(os.path.abspath(__file__))

    full_path = os.path.join(current_dir, file_path)
    # ----------------------------------------
    
    df = pd.read_csv(full_path)

    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=random_state)
    
    if 'id' in df.columns:
        df = df.drop(columns=['id'])
    
    print(f"Data loaded successfully. Shape: {df.shape}")
    return df
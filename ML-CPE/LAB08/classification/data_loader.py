import pandas as pd
import os

def load_dataset(file_path="train.csv", sample_size=5000):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), file_path)
    df = pd.read_csv(path)
    
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)
        
    return df.drop(columns=['id'], errors='ignore')
from sklearn.model_selection import train_test_split

def split_dataset(X, y, test_size=0.2):
    # แบ่ง Train 80% / Test 20%
    return train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
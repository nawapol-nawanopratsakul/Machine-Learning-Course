import pandas as pd


def load_data():
    df = pd.read_csv("car_details.csv")
    print("โหลดข้อมูลสำเร็จ! ขนาดข้อมูล:", df.shape)
    return df


if __name__ == "__main__":
    load_data()
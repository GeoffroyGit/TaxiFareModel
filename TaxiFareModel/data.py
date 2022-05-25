import pandas as pd
from google.cloud import storage
from TaxiFareModel.params import LOCAL_PATH
from TaxiFareModel.params import BUCKET_NAME, BUCKET_STORAGE_PATH, BUCKET_TRAIN_DATA_PATH

from TaxiFareModel.utils import df_optimize

def get_data(nrows=10_000):
    '''returns a DataFrame with nrows from s3 bucket'''
    #df = pd.read_csv(AWS_BUCKET_PATH, nrows=nrows)
    #df = pd.read_csv(LOCAL_PATH, nrows=nrows)
    df = pd.read_csv(f"gs://{BUCKET_NAME}/{BUCKET_TRAIN_DATA_PATH}", nrows=nrows)

    df = df_optimize(df)

    return df


def clean_data(df, test=False):
    df = df.dropna(how='any', axis='rows')
    df = df[(df.dropoff_latitude != 0) | (df.dropoff_longitude != 0)]
    df = df[(df.pickup_latitude != 0) | (df.pickup_longitude != 0)]
    if "fare_amount" in list(df):
        df = df[df.fare_amount.between(0, 4000)]
    df = df[df.passenger_count < 8]
    df = df[df.passenger_count >= 0]
    df = df[df["pickup_latitude"].between(left=40, right=42)]
    df = df[df["pickup_longitude"].between(left=-74.3, right=-72.9)]
    df = df[df["dropoff_latitude"].between(left=40, right=42)]
    df = df[df["dropoff_longitude"].between(left=-74, right=-72.9)]
    return df


def upload_model_to_gcp(filename):
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(BUCKET_STORAGE_PATH)
    blob.upload_from_filename(filename)


if __name__ == '__main__':
    df = get_data()
    df = clean_data(df)

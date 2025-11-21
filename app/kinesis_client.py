import boto3
from config import AWS_PROFILE, REGION_NAME

def get_kinesis_client():
    session = boto3.Session(profile_name=AWS_PROFILE)
    return session.client("kinesis", region_name=REGION_NAME)

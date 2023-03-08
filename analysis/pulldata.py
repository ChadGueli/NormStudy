import argparse
import boto3
import csv
import os

class GetFiles(object):

    def __init__(self, session, bucket_name):
        self.s3client = session.client("s3")
        self.bucket = session.resource("s3").Bucket(bucket_name)
        self.bucket_name = bucket_name

    def __call__(self, localpath):
        ckpt_path = os.path.join(localpath, "ckpts")
        metric_path = os.path.join(localpath, "metrics")
        for p in [ckpt_path, metric_path]:
            try:
                os.mkdir(p)
            except FileExistsError:
                pass
        
        object_lists = (self.s3client
            .get_paginator('list_objects')
            .paginate(Bucket=self.bucket_name))
        
        for object_list in object_lists:
            for obj in object_list['Contents']:
                _, ext = obj["Key"].split(".")
                self.bucket.download_file(
                    obj["Key"],
                    os.path.join(
                        localpath,
                        "ckpts" if ext=="ckpt" else "",
                        obj["Key"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--aws_cred",
        help="File path to csv containing aws credentials.",
        default="../aws_cred.csv")
    
    with open(parser.parse_args().aws_cred) as f:
            aws_creds = next(csv.DictReader(f))

    session = boto3.session.Session(
        aws_access_key_id=aws_creds["Access key ID"],
        aws_secret_access_key=aws_creds["Secret access key"])
    
    get_files = GetFiles(session, "reg-study-bucket")
    get_files("../data")

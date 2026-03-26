"""
Lambda function: Start the OCR pipeline instance on schedule.

Triggered by EventBridge daily cron. Checks if the S3 input bucket has
files before starting the instance (avoids wasting compute on empty days).

Environment variables:
    PIPELINE_INSTANCE_ID: EC2 instance ID to start
    INPUT_BUCKET: S3 bucket name
    INPUT_PREFIX: S3 prefix for input PDFs (default: "input/")
"""

import os
import boto3


def handler(event, context):
    instance_id = os.environ["PIPELINE_INSTANCE_ID"]
    bucket = os.environ["INPUT_BUCKET"]
    prefix = os.environ.get("INPUT_PREFIX", "input/")

    s3 = boto3.client("s3")
    ec2 = boto3.client("ec2")

    # Check if there are PDFs in the input prefix
    response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=1)
    if response.get("KeyCount", 0) == 0:
        print(f"No files in s3://{bucket}/{prefix}, skipping.")
        return {"status": "skipped", "reason": "no_input_files"}

    # Check instance state
    desc = ec2.describe_instances(InstanceIds=[instance_id])
    state = desc["Reservations"][0]["Instances"][0]["State"]["Name"]
    if state == "running":
        print(f"Instance {instance_id} already running, skipping start.")
        return {"status": "skipped", "reason": "already_running"}

    # Start the instance
    print(f"Starting instance {instance_id}...")
    ec2.start_instances(InstanceIds=[instance_id])

    return {"status": "started", "instance_id": instance_id}

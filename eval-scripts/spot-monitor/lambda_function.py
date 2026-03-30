"""Spot price and on-demand availability monitor for GPU instances.

Runs every 15 minutes via EventBridge. Queries all AWS regions for spot prices
and on-demand availability for configured instance types. Appends results to a
daily CSV in S3.

Environment variables:
    S3_BUCKET: S3 bucket name for storing results
    INSTANCE_TYPES: semicolon-separated instance types (default: g5.12xlarge;p4d.24xlarge;p5.4xlarge)
"""

import boto3
import csv
import io
import logging
import os
from datetime import datetime, timezone

logger = logging.getLogger()
logger.setLevel(logging.INFO)

BUCKET = os.environ["S3_BUCKET"]
INSTANCE_TYPES = os.environ.get(
    "INSTANCE_TYPES", "g5.12xlarge;p4d.24xlarge;p5.4xlarge"
).split(";")

CSV_HEADER = [
    "timestamp",
    "region",
    "instance_type",
    "spot_price_usd",
    "spot_available",
    "ondemand_available",
]


def get_all_regions():
    """Get all enabled AWS regions."""
    ec2 = boto3.client("ec2", region_name="us-east-1")
    response = ec2.describe_regions(AllRegions=False)
    return sorted(r["RegionName"] for r in response["Regions"])


def check_spot(ec2_client, instance_type):
    """Check spot price and availability for an instance type.

    Returns (price_usd: float|None, available: bool).
    """
    try:
        response = ec2_client.describe_spot_price_history(
            InstanceTypes=[instance_type],
            ProductDescriptions=["Linux/UNIX"],
            MaxResults=1,
        )
        prices = response.get("SpotPriceHistory", [])
        if prices:
            return float(prices[0]["SpotPrice"]), True
        return None, False
    except Exception as e:
        logger.warning(f"Spot check failed for {instance_type}: {e}")
        return None, False


def resolve_ami(ssm_client):
    """Resolve latest Amazon Linux 2023 AMI ID via SSM public parameter."""
    try:
        response = ssm_client.get_parameter(
            Name="/aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-x86_64"
        )
        return response["Parameter"]["Value"]
    except Exception as e:
        logger.warning(f"AMI resolution failed: {e}")
        return None


def check_ondemand(ec2_client, instance_type, ami_id):
    """Check on-demand availability via dry-run RunInstances.

    Returns True if capacity is available, False otherwise.
    """
    if not ami_id:
        return None

    try:
        ec2_client.run_instances(
            InstanceType=instance_type,
            ImageId=ami_id,
            MinCount=1,
            MaxCount=1,
            DryRun=True,
        )
        # Should never reach here — DryRun always raises
        return True
    except ec2_client.exceptions.ClientError as e:
        error_code = e.response["Error"]["Code"]
        if error_code == "DryRunOperation":
            # "would have succeeded" — capacity available
            return True
        elif error_code in (
            "InsufficientInstanceCapacity",
            "InstanceLimitExceeded",
            "Unsupported",
        ):
            return False
        elif error_code == "InvalidParameterValue":
            # Instance type not offered in this region
            return False
        else:
            logger.warning(
                f"On-demand check unexpected error for {instance_type}: {error_code} — {e}"
            )
            return False
    except Exception as e:
        logger.warning(f"On-demand check failed for {instance_type}: {e}")
        return False


def append_to_s3_csv(s3_key, rows):
    """Append rows to a daily CSV in S3. Creates the file if it doesn't exist."""
    s3 = boto3.client("s3", region_name="eu-west-1")

    # Try to read existing file
    existing_content = ""
    try:
        response = s3.get_object(Bucket=BUCKET, Key=s3_key)
        existing_content = response["Body"].read().decode("utf-8")
    except s3.exceptions.NoSuchKey:
        pass
    except Exception as e:
        logger.warning(f"Failed to read existing CSV: {e}")

    # Build new content
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=CSV_HEADER)

    if not existing_content:
        writer.writeheader()
    else:
        output.write(existing_content)
        if not existing_content.endswith("\n"):
            output.write("\n")

    for row in rows:
        writer.writerow(row)

    s3.put_object(
        Bucket=BUCKET,
        Key=s3_key,
        Body=output.getvalue().encode("utf-8"),
        ContentType="text/csv",
    )
    logger.info(f"Wrote {len(rows)} rows to s3://{BUCKET}/{s3_key}")


def lambda_handler(event, context):
    now = datetime.now(timezone.utc)
    timestamp = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    date_key = now.strftime("%Y-%m-%d")
    s3_key = f"spot-monitor/{date_key}.csv"

    regions = get_all_regions()
    logger.info(f"Monitoring {len(INSTANCE_TYPES)} instance types across {len(regions)} regions")

    rows = []
    for region in regions:
        ec2 = boto3.client("ec2", region_name=region)
        ssm = boto3.client("ssm", region_name=region)
        ami_id = resolve_ami(ssm)

        for itype in INSTANCE_TYPES:
            spot_price, spot_available = check_spot(ec2, itype)
            ondemand_available = check_ondemand(ec2, itype, ami_id)

            rows.append({
                "timestamp": timestamp,
                "region": region,
                "instance_type": itype,
                "spot_price_usd": spot_price if spot_price is not None else "",
                "spot_available": spot_available,
                "ondemand_available": ondemand_available if ondemand_available is not None else "",
            })

    append_to_s3_csv(s3_key, rows)

    logger.info(f"Completed: {len(rows)} records collected")
    return {
        "statusCode": 200,
        "rows": len(rows),
        "timestamp": timestamp,
    }

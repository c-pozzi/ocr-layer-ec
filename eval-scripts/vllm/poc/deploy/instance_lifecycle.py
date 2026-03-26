#!/usr/bin/env python3
"""
EC2 instance lifecycle: completion signaling and self-shutdown.
"""

import logging
import subprocess

log = logging.getLogger(__name__)


def get_instance_id() -> str | None:
    """Retrieve this instance's ID from EC2 metadata (IMDSv2)."""
    try:
        token = subprocess.check_output(
            ["curl", "-s", "-X", "PUT",
             "http://169.254.169.254/latest/api/token",
             "-H", "X-aws-ec2-metadata-token-ttl-seconds: 60"],
            timeout=5,
        ).decode().strip()
        instance_id = subprocess.check_output(
            ["curl", "-s",
             "http://169.254.169.254/latest/meta-data/instance-id",
             "-H", f"X-aws-ec2-metadata-token: {token}"],
            timeout=5,
        ).decode().strip()
        return instance_id
    except Exception as e:
        log.warning("Could not get instance ID: %s", e)
        return None


def shutdown_instance() -> None:
    """
    Stop (not terminate) this EC2 instance.

    Falls back to ``sudo shutdown -h +1`` if the EC2 API call fails.
    """
    instance_id = get_instance_id()

    if instance_id:
        try:
            import boto3
            ec2 = boto3.client("ec2")
            log.info("Stopping instance %s via EC2 API", instance_id)
            ec2.stop_instances(InstanceIds=[instance_id])
            return
        except Exception as e:
            log.warning("EC2 stop_instances failed: %s, falling back to shutdown", e)

    log.info("Issuing shutdown -h +1")
    subprocess.run(["sudo", "shutdown", "-h", "+1"], check=False)

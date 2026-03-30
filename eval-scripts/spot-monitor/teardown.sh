#!/usr/bin/env bash
# Remove all spot monitor AWS resources.
# Usage: bash teardown.sh
set -euo pipefail

REGION="eu-west-1"
ACCOUNT_ID="991345721932"
BUCKET="ocr-spot-monitor-${ACCOUNT_ID}"
FUNCTION_NAME="spot-price-monitor"
ROLE_NAME="spot-monitor-lambda-role"
RULE_NAME="spot-monitor-every-15min"

echo "=== Tearing down Spot Price Monitor ==="
echo ""

# 1. Remove EventBridge
echo "[1/4] Removing EventBridge rule..."
aws events remove-targets --rule "${RULE_NAME}" --ids "1" --region "${REGION}" 2>/dev/null || true
aws events delete-rule --name "${RULE_NAME}" --region "${REGION}" 2>/dev/null || true
echo "  Done."

# 2. Delete Lambda
echo "[2/4] Deleting Lambda function..."
aws lambda delete-function --function-name "${FUNCTION_NAME}" --region "${REGION}" 2>/dev/null || true
echo "  Done."

# 3. Delete IAM role
echo "[3/4] Deleting IAM role..."
aws iam delete-role-policy --role-name "${ROLE_NAME}" --policy-name "spot-monitor-policy" 2>/dev/null || true
aws iam delete-role --role-name "${ROLE_NAME}" 2>/dev/null || true
echo "  Done."

# 4. S3 bucket (ask first)
echo "[4/4] S3 bucket s3://${BUCKET}"
read -p "  Delete S3 bucket and all data? (y/N): " confirm
if [[ "${confirm}" =~ ^[Yy]$ ]]; then
    aws s3 rb "s3://${BUCKET}" --force --region "${REGION}"
    echo "  Bucket deleted."
else
    echo "  Bucket kept."
fi

echo ""
echo "=== Teardown complete ==="

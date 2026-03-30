#!/usr/bin/env bash
# Deploy spot price monitor: S3 bucket, IAM role, Lambda, EventBridge rule.
# Usage: bash deploy.sh
set -euo pipefail

REGION="eu-west-1"
ACCOUNT_ID="991345721932"
BUCKET="ocr-spot-monitor-${ACCOUNT_ID}"
FUNCTION_NAME="spot-price-monitor"
ROLE_NAME="spot-monitor-lambda-role"
RULE_NAME="spot-monitor-every-15min"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== Deploying Spot Price Monitor ==="
echo "Region: ${REGION}"
echo "Bucket: ${BUCKET}"
echo ""

# -----------------------------------------------------------------------
# 1. S3 Bucket
# -----------------------------------------------------------------------
echo "[1/5] Creating S3 bucket..."
if aws s3api head-bucket --bucket "${BUCKET}" 2>/dev/null; then
    echo "  Bucket already exists, skipping."
else
    aws s3 mb "s3://${BUCKET}" --region "${REGION}"
    echo "  Created bucket s3://${BUCKET}"
fi

# Add lifecycle rule to expire data after 90 days
aws s3api put-bucket-lifecycle-configuration \
    --bucket "${BUCKET}" \
    --lifecycle-configuration '{
        "Rules": [{
            "ID": "expire-90d",
            "Status": "Enabled",
            "Filter": {"Prefix": "spot-monitor/"},
            "Expiration": {"Days": 90}
        }]
    }'
echo "  Lifecycle rule set (90-day expiry)."

# -----------------------------------------------------------------------
# 2. IAM Role
# -----------------------------------------------------------------------
echo "[2/5] Creating IAM role..."

TRUST_POLICY='{
    "Version": "2012-10-17",
    "Statement": [{
        "Effect": "Allow",
        "Principal": {"Service": "lambda.amazonaws.com"},
        "Action": "sts:AssumeRole"
    }]
}'

if aws iam get-role --role-name "${ROLE_NAME}" 2>/dev/null; then
    echo "  Role already exists, updating policy."
else
    aws iam create-role \
        --role-name "${ROLE_NAME}" \
        --assume-role-policy-document "${TRUST_POLICY}" \
        --description "Spot price monitor Lambda execution role" \
        > /dev/null
    echo "  Created role ${ROLE_NAME}"
    echo "  Waiting 10s for role propagation..."
    sleep 10
fi

# Inline policy
POLICY='{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "EC2ReadOnly",
            "Effect": "Allow",
            "Action": [
                "ec2:DescribeSpotPriceHistory",
                "ec2:DescribeRegions",
                "ec2:RunInstances"
            ],
            "Resource": "*"
        },
        {
            "Sid": "DenyActualLaunch",
            "Effect": "Deny",
            "Action": "ec2:RunInstances",
            "Resource": "arn:aws:ec2:*:*:instance/*"
        },
        {
            "Sid": "SSMGetAmi",
            "Effect": "Allow",
            "Action": "ssm:GetParameter",
            "Resource": "arn:aws:ssm:*:*:parameter/aws/service/ami-amazon-linux-latest/*"
        },
        {
            "Sid": "S3ReadWrite",
            "Effect": "Allow",
            "Action": ["s3:GetObject", "s3:PutObject"],
            "Resource": "arn:aws:s3:::'"${BUCKET}"'/spot-monitor/*"
        },
        {
            "Sid": "CloudWatchLogs",
            "Effect": "Allow",
            "Action": [
                "logs:CreateLogGroup",
                "logs:CreateLogStream",
                "logs:PutLogEvents"
            ],
            "Resource": "arn:aws:logs:*:'"${ACCOUNT_ID}"':*"
        }
    ]
}'

aws iam put-role-policy \
    --role-name "${ROLE_NAME}" \
    --policy-name "spot-monitor-policy" \
    --policy-document "${POLICY}"
echo "  Attached inline policy."

ROLE_ARN="arn:aws:iam::${ACCOUNT_ID}:role/${ROLE_NAME}"

# -----------------------------------------------------------------------
# 3. Lambda Function
# -----------------------------------------------------------------------
echo "[3/5] Deploying Lambda function..."

cd "${SCRIPT_DIR}"
python3 -c "import zipfile; z=zipfile.ZipFile('lambda_function.zip','w'); z.write('lambda_function.py'); z.close()"

if aws lambda get-function --function-name "${FUNCTION_NAME}" --region "${REGION}" 2>/dev/null; then
    echo "  Function exists, updating code..."
    aws lambda update-function-code \
        --function-name "${FUNCTION_NAME}" \
        --zip-file fileb://lambda_function.zip \
        --region "${REGION}" > /dev/null
    # Wait for update to complete before updating config
    aws lambda wait function-updated --function-name "${FUNCTION_NAME}" --region "${REGION}"
    aws lambda update-function-configuration \
        --function-name "${FUNCTION_NAME}" \
        --timeout 120 \
        --memory-size 256 \
        --environment "Variables={S3_BUCKET=${BUCKET},INSTANCE_TYPES=g5.12xlarge;p4d.24xlarge;p5.4xlarge}" \
        --region "${REGION}" > /dev/null
else
    aws lambda create-function \
        --function-name "${FUNCTION_NAME}" \
        --runtime python3.12 \
        --handler lambda_function.lambda_handler \
        --role "${ROLE_ARN}" \
        --zip-file fileb://lambda_function.zip \
        --timeout 120 \
        --memory-size 256 \
        --environment "Variables={S3_BUCKET=${BUCKET},INSTANCE_TYPES=g5.12xlarge;p4d.24xlarge;p5.4xlarge}" \
        --region "${REGION}" > /dev/null
    echo "  Created function ${FUNCTION_NAME}"
fi

rm -f lambda_function.zip

# Wait for function to be active
echo "  Waiting for function to be active..."
aws lambda wait function-active-v2 --function-name "${FUNCTION_NAME}" --region "${REGION}"
echo "  Lambda is active."

# -----------------------------------------------------------------------
# 4. EventBridge Rule (every 15 minutes)
# -----------------------------------------------------------------------
echo "[4/5] Creating EventBridge rule..."

aws events put-rule \
    --name "${RULE_NAME}" \
    --schedule-expression "rate(15 minutes)" \
    --state ENABLED \
    --description "Trigger spot price monitor every 15 minutes" \
    --region "${REGION}" > /dev/null

FUNCTION_ARN="arn:aws:lambda:${REGION}:${ACCOUNT_ID}:function:${FUNCTION_NAME}"

aws events put-targets \
    --rule "${RULE_NAME}" \
    --targets "Id=1,Arn=${FUNCTION_ARN}" \
    --region "${REGION}" > /dev/null

# Grant EventBridge permission to invoke Lambda
aws lambda add-permission \
    --function-name "${FUNCTION_NAME}" \
    --statement-id eventbridge-invoke \
    --action lambda:InvokeFunction \
    --principal events.amazonaws.com \
    --source-arn "arn:aws:events:${REGION}:${ACCOUNT_ID}:rule/${RULE_NAME}" \
    --region "${REGION}" 2>/dev/null || true  # ignore if already exists

echo "  EventBridge rule created: every 15 minutes."

# -----------------------------------------------------------------------
# 5. Test invocation
# -----------------------------------------------------------------------
echo "[5/5] Running test invocation..."

aws lambda invoke \
    --function-name "${FUNCTION_NAME}" \
    --region "${REGION}" \
    --payload '{}' \
    /tmp/spot-monitor-test-output.json > /dev/null

echo "  Test result:"
cat /tmp/spot-monitor-test-output.json
echo ""
rm -f /tmp/spot-monitor-test-output.json

echo ""
echo "=== Deployment complete ==="
echo "Monitor data: s3://${BUCKET}/spot-monitor/"
echo "Check logs:   aws logs tail /aws/lambda/${FUNCTION_NAME} --region ${REGION}"
echo "Check data:   aws s3 cp s3://${BUCKET}/spot-monitor/$(date -u +%Y-%m-%d).csv -"

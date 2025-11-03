"""
Machine Learning Workload Tools

Provides LangChain tools for AWS ML services including SageMaker,
Amazon Bedrock, and cost optimization for ML workloads.
"""

import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, List

from langchain_core.tools import tool

from .aws_helpers import _get_cross_account_client, _format_account_context

logger = logging.getLogger(__name__)


@tool
def list_sagemaker_training_jobs(
    account_id: Optional[str] = None,
    role_name: Optional[str] = None,
    days: int = 7,
    max_results: int = 50,
) -> str:
    """
    List recent SageMaker training jobs in an AWS account.

    Use this tool to discover training jobs and their status for cost analysis.

    Args:
        account_id: Target AWS account ID for cross-account access (optional)
        role_name: IAM role name to assume in target account (optional)
        days: Number of days to look back for training jobs (default: 7)
        max_results: Maximum number of training jobs to return (default: 50)

    Returns:
        Formatted string with list of training jobs and their details
    """
    try:
        sagemaker = _get_cross_account_client("sagemaker", account_id, role_name)
        account_context = _format_account_context(account_id)

        # Calculate start time
        creation_time_after = datetime.now() - timedelta(days=days)

        response = sagemaker.list_training_jobs(
            CreationTimeAfter=creation_time_after,
            MaxResults=max_results,
            SortBy="CreationTime",
            SortOrder="Descending",
        )

        training_jobs = response.get("TrainingJobSummaries", [])

        if not training_jobs:
            return f"No SageMaker training jobs found in the last {days} days in {account_context}."

        result = [
            f"Found {len(training_jobs)} SageMaker training job(s) in the last {days} days in {account_context}:\n"
        ]

        for job in training_jobs:
            job_name = job["TrainingJobName"]
            status = job["TrainingJobStatus"]
            creation_time = job["CreationTime"].strftime("%Y-%m-%d %H:%M:%S")
            instance_type = job.get("ResourceConfig", {}).get("InstanceType", "N/A")
            instance_count = job.get("ResourceConfig", {}).get("InstanceCount", "N/A")

            duration = "N/A"
            if "TrainingEndTime" in job:
                duration_seconds = (
                    job["TrainingEndTime"] - job["CreationTime"]
                ).total_seconds()
                duration = f"{duration_seconds / 3600:.2f} hours"

            result.append(f"  - {job_name}")
            result.append(f"    Status: {status}")
            result.append(f"    Created: {creation_time}")
            result.append(f"    Instance: {instance_type} (Count: {instance_count})")
            result.append(f"    Duration: {duration}\n")

        logger.info(f"Listed {len(training_jobs)} training jobs from {account_context}")
        return "\n".join(result)

    except Exception as e:
        error_msg = f"Error listing SageMaker training jobs: {str(e)}"
        logger.error(error_msg)
        return error_msg


@tool
def get_training_job_details(
    training_job_name: str,
    account_id: Optional[str] = None,
    role_name: Optional[str] = None,
) -> str:
    """
    Get detailed information about a specific SageMaker training job.

    Use this tool to analyze training job configuration and costs.

    Args:
        training_job_name: Name of the SageMaker training job
        account_id: Target AWS account ID for cross-account access (optional)
        role_name: IAM role name to assume in target account (optional)

    Returns:
        Formatted string with detailed training job information
    """
    try:
        sagemaker = _get_cross_account_client("sagemaker", account_id, role_name)
        account_context = _format_account_context(account_id)

        job = sagemaker.describe_training_job(TrainingJobName=training_job_name)

        # Calculate duration and estimate cost
        duration_seconds = 0
        if "TrainingEndTime" in job:
            duration_seconds = (
                job["TrainingEndTime"] - job["CreationTime"]
            ).total_seconds()
        elif job["TrainingJobStatus"] == "InProgress":
            duration_seconds = (
                datetime.now() - job["CreationTime"].replace(tzinfo=None)
            ).total_seconds()

        duration_hours = duration_seconds / 3600

        result = [
            f"Training Job: {training_job_name}",
            f"Account: {account_context}",
            f"Status: {job['TrainingJobStatus']}",
            f"\nResource Configuration:",
            f"  Instance Type: {job['ResourceConfig']['InstanceType']}",
            f"  Instance Count: {job['ResourceConfig']['InstanceCount']}",
            f"  Volume Size: {job['ResourceConfig']['VolumeSizeInGB']} GB",
            f"\nTiming:",
            f"  Created: {job['CreationTime'].strftime('%Y-%m-%d %H:%M:%S')}",
        ]

        if "TrainingEndTime" in job:
            result.append(
                f"  Ended: {job['TrainingEndTime'].strftime('%Y-%m-%d %H:%M:%S')}"
            )
        result.append(f"  Duration: {duration_hours:.2f} hours")

        if "BillableTimeInSeconds" in job:
            billable_hours = job["BillableTimeInSeconds"] / 3600
            result.append(f"  Billable Time: {billable_hours:.2f} hours")

        if "FinalMetricDataList" in job and job["FinalMetricDataList"]:
            result.append(f"\nFinal Metrics:")
            for metric in job["FinalMetricDataList"]:
                result.append(f"  {metric['MetricName']}: {metric['Value']}")

        logger.info(f"Retrieved details for training job {training_job_name}")
        return "\n".join(result)

    except Exception as e:
        error_msg = f"Error getting training job details for '{training_job_name}': {str(e)}"
        logger.error(error_msg)
        return error_msg


@tool
def list_sagemaker_endpoints(
    account_id: Optional[str] = None,
    role_name: Optional[str] = None,
    max_results: int = 50,
) -> str:
    """
    List SageMaker endpoints in an AWS account.

    Use this tool to discover active inference endpoints for cost analysis.

    Args:
        account_id: Target AWS account ID for cross-account access (optional)
        role_name: IAM role name to assume in target account (optional)
        max_results: Maximum number of endpoints to return (default: 50)

    Returns:
        Formatted string with list of endpoints and their status
    """
    try:
        sagemaker = _get_cross_account_client("sagemaker", account_id, role_name)
        account_context = _format_account_context(account_id)

        response = sagemaker.list_endpoints(
            MaxResults=max_results,
            SortBy="CreationTime",
            SortOrder="Descending",
        )

        endpoints = response.get("Endpoints", [])

        if not endpoints:
            return f"No SageMaker endpoints found in {account_context}."

        result = [f"Found {len(endpoints)} SageMaker endpoint(s) in {account_context}:\n"]

        for endpoint in endpoints:
            endpoint_name = endpoint["EndpointName"]
            status = endpoint["EndpointStatus"]
            creation_time = endpoint["CreationTime"].strftime("%Y-%m-%d %H:%M:%S")

            result.append(f"  - {endpoint_name}")
            result.append(f"    Status: {status}")
            result.append(f"    Created: {creation_time}\n")

        logger.info(f"Listed {len(endpoints)} endpoints from {account_context}")
        return "\n".join(result)

    except Exception as e:
        error_msg = f"Error listing SageMaker endpoints: {str(e)}"
        logger.error(error_msg)
        return error_msg


@tool
def get_endpoint_details(
    endpoint_name: str,
    account_id: Optional[str] = None,
    role_name: Optional[str] = None,
) -> str:
    """
    Get detailed information about a specific SageMaker endpoint.

    Use this tool to analyze endpoint configuration and costs.

    Args:
        endpoint_name: Name of the SageMaker endpoint
        account_id: Target AWS account ID for cross-account access (optional)
        role_name: IAM role name to assume in target account (optional)

    Returns:
        Formatted string with detailed endpoint information
    """
    try:
        sagemaker = _get_cross_account_client("sagemaker", account_id, role_name)
        cloudwatch = _get_cross_account_client("cloudwatch", account_id, role_name)
        account_context = _format_account_context(account_id)

        endpoint = sagemaker.describe_endpoint(EndpointName=endpoint_name)

        result = [
            f"Endpoint: {endpoint_name}",
            f"Account: {account_context}",
            f"Status: {endpoint['EndpointStatus']}",
            f"Created: {endpoint['CreationTime'].strftime('%Y-%m-%d %H:%M:%S')}",
            f"\nEndpoint Configuration:",
        ]

        # Get endpoint config details
        config_name = endpoint["EndpointConfigName"]
        config = sagemaker.describe_endpoint_config(EndpointConfigName=config_name)

        for variant in config["ProductionVariants"]:
            result.append(f"\n  Variant: {variant['VariantName']}")
            result.append(f"    Instance Type: {variant['InstanceType']}")
            result.append(f"    Instance Count: {variant['InitialInstanceCount']}")
            result.append(f"    Model: {variant['ModelName']}")

        # Get recent invocation metrics
        try:
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=24)

            metrics = cloudwatch.get_metric_statistics(
                Namespace="AWS/SageMaker",
                MetricName="Invocations",
                Dimensions=[
                    {"Name": "EndpointName", "Value": endpoint_name},
                    {"Name": "VariantName", "Value": "AllTraffic"},
                ],
                StartTime=start_time,
                EndTime=end_time,
                Period=86400,  # 24 hours
                Statistics=["Sum"],
            )

            if metrics["Datapoints"]:
                total_invocations = sum(
                    point["Sum"] for point in metrics["Datapoints"]
                )
                result.append(
                    f"\n  Invocations (last 24h): {int(total_invocations):,}"
                )

        except Exception as metric_error:
            logger.warning(f"Could not retrieve metrics: {str(metric_error)}")

        logger.info(f"Retrieved details for endpoint {endpoint_name}")
        return "\n".join(result)

    except Exception as e:
        error_msg = f"Error getting endpoint details for '{endpoint_name}': {str(e)}"
        logger.error(error_msg)
        return error_msg


@tool
def analyze_bedrock_usage(
    account_id: Optional[str] = None,
    role_name: Optional[str] = None,
    days: int = 7,
) -> str:
    """
    Analyze Amazon Bedrock model usage and costs.

    Use this tool to understand Bedrock inference costs and usage patterns.

    Args:
        account_id: Target AWS account ID for cross-account access (optional)
        role_name: IAM role name to assume in target account (optional)
        days: Number of days to analyze (default: 7)

    Returns:
        Formatted string with Bedrock usage analysis
    """
    try:
        cloudwatch = _get_cross_account_client("cloudwatch", account_id, role_name)
        account_context = _format_account_context(account_id)

        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)

        # Get invocation metrics
        invocations_metrics = cloudwatch.get_metric_statistics(
            Namespace="AWS/Bedrock",
            MetricName="Invocations",
            StartTime=start_time,
            EndTime=end_time,
            Period=86400,  # Daily
            Statistics=["Sum"],
        )

        # Get input/output token metrics if available
        input_tokens_metrics = cloudwatch.get_metric_statistics(
            Namespace="AWS/Bedrock",
            MetricName="InputTokens",
            StartTime=start_time,
            EndTime=end_time,
            Period=86400,
            Statistics=["Sum"],
        )

        output_tokens_metrics = cloudwatch.get_metric_statistics(
            Namespace="AWS/Bedrock",
            MetricName="OutputTokens",
            StartTime=start_time,
            EndTime=end_time,
            Period=86400,
            Statistics=["Sum"],
        )

        result = [
            f"Amazon Bedrock Usage Analysis",
            f"Account: {account_context}",
            f"Period: Last {days} days\n",
        ]

        total_invocations = sum(
            point["Sum"] for point in invocations_metrics.get("Datapoints", [])
        )
        total_input_tokens = sum(
            point["Sum"] for point in input_tokens_metrics.get("Datapoints", [])
        )
        total_output_tokens = sum(
            point["Sum"] for point in output_tokens_metrics.get("Datapoints", [])
        )

        result.append(f"Total Invocations: {int(total_invocations):,}")
        result.append(f"Total Input Tokens: {int(total_input_tokens):,}")
        result.append(f"Total Output Tokens: {int(total_output_tokens):,}")
        result.append(
            f"Average Tokens per Request: {int((total_input_tokens + total_output_tokens) / max(total_invocations, 1)):,}"
        )

        logger.info(
            f"Analyzed Bedrock usage for {days} days in {account_context}: {int(total_invocations)} invocations"
        )
        return "\n".join(result)

    except Exception as e:
        error_msg = f"Error analyzing Bedrock usage: {str(e)}"
        logger.error(error_msg)
        return error_msg


@tool
def get_ml_cost_recommendations(
    account_id: Optional[str] = None,
    role_name: Optional[str] = None,
) -> str:
    """
    Get AWS Cost Explorer recommendations for ML workloads.

    Use this tool to identify cost optimization opportunities for ML services.

    Args:
        account_id: Target AWS account ID for cross-account access (optional)
        role_name: IAM role name to assume in target account (optional)

    Returns:
        Formatted string with cost recommendations
    """
    try:
        ce_client = _get_cross_account_client("ce", account_id, role_name)
        account_context = _format_account_context(account_id)

        # Get cost for ML services in the last 30 days
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

        ml_services = ["Amazon SageMaker", "Amazon Bedrock", "AWS Deep Learning"]

        result = [
            f"ML Cost Analysis and Recommendations",
            f"Account: {account_context}",
            f"Period: Last 30 days\n",
        ]

        for service in ml_services:
            try:
                response = ce_client.get_cost_and_usage(
                    TimePeriod={"Start": start_date, "End": end_date},
                    Granularity="MONTHLY",
                    Filter={
                        "Dimensions": {
                            "Key": "SERVICE",
                            "Values": [service],
                        }
                    },
                    Metrics=["UnblendedCost"],
                )

                total_cost = 0
                for period in response.get("ResultsByTime", []):
                    total_cost += float(
                        period["Total"]["UnblendedCost"].get("Amount", 0)
                    )

                if total_cost > 0:
                    result.append(f"{service}: ${total_cost:.2f}")

            except Exception as service_error:
                logger.warning(
                    f"Could not get cost for {service}: {str(service_error)}"
                )

        # Add general recommendations
        result.append(f"\nCost Optimization Recommendations:")
        result.append(
            f"  1. Review idle SageMaker endpoints and consider auto-scaling"
        )
        result.append(f"  2. Use Spot instances for non-critical training jobs")
        result.append(
            f"  3. Implement model caching for Bedrock to reduce token usage"
        )
        result.append(
            f"  4. Consider using SageMaker Savings Plans for predictable workloads"
        )
        result.append(f"  5. Clean up unused models and endpoint configurations")

        logger.info(f"Generated ML cost recommendations for {account_context}")
        return "\n".join(result)

    except Exception as e:
        error_msg = f"Error getting ML cost recommendations: {str(e)}"
        logger.error(error_msg)
        return error_msg


@tool
def analyze_ml_data_storage(
    account_id: Optional[str] = None,
    role_name: Optional[str] = None,
) -> str:
    """
    Analyze S3 storage costs for ML training data and models.

    Use this tool to identify opportunities to optimize data storage costs.

    Args:
        account_id: Target AWS account ID for cross-account access (optional)
        role_name: IAM role name to assume in target account (optional)

    Returns:
        Formatted string with storage analysis
    """
    try:
        s3_client = _get_cross_account_client("s3", account_id, role_name)
        cloudwatch = _get_cross_account_client("cloudwatch", account_id, role_name)
        account_context = _format_account_context(account_id)

        result = [
            f"ML Data Storage Analysis",
            f"Account: {account_context}\n",
        ]

        # List buckets that might contain ML data
        buckets_response = s3_client.list_buckets()
        ml_buckets = []

        for bucket in buckets_response.get("Buckets", []):
            bucket_name = bucket["Name"]
            # Look for common ML-related naming patterns
            if any(
                pattern in bucket_name.lower()
                for pattern in ["sagemaker", "ml", "model", "training", "dataset"]
            ):
                ml_buckets.append(bucket_name)

        if not ml_buckets:
            return f"No ML-related S3 buckets found in {account_context}."

        result.append(f"Found {len(ml_buckets)} ML-related bucket(s):\n")

        for bucket_name in ml_buckets[:10]:  # Limit to 10 buckets
            try:
                # Get bucket size from CloudWatch
                end_time = datetime.now()
                start_time = end_time - timedelta(days=1)

                size_metrics = cloudwatch.get_metric_statistics(
                    Namespace="AWS/S3",
                    MetricName="BucketSizeBytes",
                    Dimensions=[
                        {"Name": "BucketName", "Value": bucket_name},
                        {"Name": "StorageType", "Value": "StandardStorage"},
                    ],
                    StartTime=start_time,
                    EndTime=end_time,
                    Period=86400,
                    Statistics=["Average"],
                )

                if size_metrics["Datapoints"]:
                    size_bytes = size_metrics["Datapoints"][0]["Average"]
                    size_gb = size_bytes / (1024**3)
                    result.append(f"  - {bucket_name}: {size_gb:.2f} GB")

            except Exception as bucket_error:
                logger.warning(
                    f"Could not get size for bucket {bucket_name}: {str(bucket_error)}"
                )
                result.append(f"  - {bucket_name}: Size unavailable")

        result.append(f"\nStorage Optimization Recommendations:")
        result.append(f"  1. Implement S3 Intelligent-Tiering for training data")
        result.append(f"  2. Set lifecycle policies to archive old training datasets")
        result.append(f"  3. Delete temporary data and failed training outputs")
        result.append(f"  4. Use S3 Standard-IA for infrequently accessed models")

        logger.info(f"Analyzed ML data storage for {len(ml_buckets)} buckets")
        return "\n".join(result)

    except Exception as e:
        error_msg = f"Error analyzing ML data storage: {str(e)}"
        logger.error(error_msg)
        return error_msg

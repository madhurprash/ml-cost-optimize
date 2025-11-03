"""
Tools Package
"""

# CloudWatch Monitoring Tools
from .cloudwatch_tools import (
    list_cloudwatch_dashboards,
    get_dashboard_summary,
    list_log_groups,
    fetch_cloudwatch_logs_for_service,
    analyze_log_group,
    get_cloudwatch_alarms_for_service,
    setup_cross_account_access,
)

# Machine Learning Tools
from .ml_tools import (
    list_sagemaker_training_jobs,
    get_training_job_details,
    list_sagemaker_endpoints,
    get_endpoint_details,
    analyze_bedrock_usage,
    get_ml_cost_recommendations,
    analyze_ml_data_storage,
)


__all__ = [
    # CloudWatch tools
    "list_cloudwatch_dashboards",
    "get_dashboard_summary",
    "list_log_groups",
    "fetch_cloudwatch_logs_for_service",
    "analyze_log_group",
    "get_cloudwatch_alarms_for_service",
    "setup_cross_account_access",
    # ML tools
    "list_sagemaker_training_jobs",
    "get_training_job_details",
    "list_sagemaker_endpoints",
    "get_endpoint_details",
    "analyze_bedrock_usage",
    "get_ml_cost_recommendations",
    "analyze_ml_data_storage",
]
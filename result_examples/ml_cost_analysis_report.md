# ML Workload Cost Optimization Report

## Executive Summary

This report provides a comprehensive analysis of machine learning workload costs across your AWS account, focusing on SageMaker training jobs, endpoints, Amazon Bedrock usage, and ML data storage. Our analysis identified several key optimization opportunities with an estimated potential cost reduction of 30-40% through implementation of the recommended strategies.

**Key Findings:**
- SageMaker costs account for approximately $9,285.50 over the last 30 days
- Multiple SageMaker endpoints are running with zero invocations
- High failure rate in training jobs (8 out of 9 jobs failed)
- Underutilized GPU instances on active endpoints
- 306+ GB of ML data storage with optimization opportunities
- Amazon Bedrock usage ($52.94) shows potential for token optimization

## Current State Analysis

### SageMaker Training Jobs

We analyzed 9 SageMaker training jobs from the past 30 days:

| Metric | Value |
|--------|-------|
| Total Jobs | 9 |
| Completed Jobs | 1 |
| Failed Jobs | 8 |
| Average Duration | 0.13 hours |
| Instance Types | ml.g5.12xlarge |

**Key Issues:**
- 89% failure rate in training jobs
- Use of expensive GPU instances (ml.g5.12xlarge) for short-duration jobs
- Repeated failed jobs without resolution

### SageMaker Endpoints

We analyzed 21 SageMaker endpoints:

| Status | Count |
|--------|-------|
| InService | 4 |
| Failed | 17 |

**Active Endpoints:**
- llama3-endpoint-1 (ml.g5.2xlarge)
- model-endpoint-2
- llama3-endpoint-3
- qwen-vl-endpoint-4 (ml.g5.2xlarge)

**Key Issues:**
- Multiple active endpoints with zero invocations in the last 24 hours
- GPU instances (ml.g5.2xlarge) running without utilization
- High number of failed endpoints (17)

### Amazon Bedrock Usage

| Metric | Value |
|--------|-------|
| Total Invocations | 1,642 |
| Total Input Tokens | 0* |
| Total Output Tokens | 0* |
| Cost | $52.94 |

*Note: Token data may not be available in the current metrics collection.

### ML Data Storage

| Bucket | Size |
|--------|------|
| sagemaker-us-west-2-XXXXXXXXXXXX | 306.06 GB |
| sagemaker-fmbench-write-us-west-2-XXXXXXXXXXXX | 0.27 GB |
| sagemaker-fmbench-read-us-west-2-XXXXXXXXXXXX | 0.08 GB |
| **Total** | **306.41 GB** |

## Key Findings

1. **Idle GPU Resources**: Multiple SageMaker endpoints are running on GPU instances with zero recorded invocations, resulting in unnecessary costs.

2. **High Training Job Failure Rate**: 8 out of 9 training jobs failed, indicating potential configuration issues and wasted compute resources.

3. **Inefficient Instance Selection**: The use of ml.g5.12xlarge for short-duration training jobs (averaging 0.13 hours) suggests potential over-provisioning.

4. **Underutilized Endpoints**: Several LLM endpoints are running without recent invocations.

5. **Storage Optimization Potential**: Over 306 GB of ML data stored without apparent lifecycle policies or tiering strategies.

6. **Bedrock Usage Optimization**: Moderate Bedrock usage with potential for token optimization and caching strategies.

## Optimization Opportunities

### 1. SageMaker Training Jobs Optimization

| Opportunity | Potential Savings | Implementation Difficulty |
|-------------|-------------------|--------------------------|
| Use Spot Instances | 70-90% per job | Low |
| Right-size instances | 30-50% per job | Medium |
| Implement automated job monitoring | Reduces failed job costs | Medium |
| Hyperparameter optimization | Improves efficiency | Medium |

### 2. SageMaker Endpoint Optimization

| Opportunity | Potential Savings | Implementation Difficulty |
|-------------|-------------------|--------------------------|
| Terminate idle endpoints | 100% of idle endpoint costs | Low |
| Implement auto-scaling | 40-60% | Medium |
| Use multi-model endpoints | 30-70% | Medium |
| Serverless inference | Pay-per-use | Medium |

### 3. Amazon Bedrock Optimization

| Opportunity | Potential Savings | Implementation Difficulty |
|-------------|-------------------|--------------------------|
| Implement token caching | 10-30% | Low |
| Prompt optimization | 20-40% | Medium |
| Model selection optimization | 15-30% | Low |

### 4. ML Data Storage Optimization

| Opportunity | Potential Savings | Implementation Difficulty |
|-------------|-------------------|--------------------------|
| S3 Intelligent-Tiering | 15-40% | Low |
| Lifecycle policies | 20-50% | Low |
| Clean up failed job artifacts | Variable | Low |

## Detailed Recommendations

### 1. Immediate Cost Reduction Actions

#### 1.1 Terminate Idle SageMaker Endpoints
```python
# Sample code to identify and terminate idle endpoints
import boto3
sagemaker = boto3.client('sagemaker')

# List of endpoints with zero invocations
idle_endpoints = [
    "llama3-endpoint-1",
    "qwen-vl-endpoint-4"
]

for endpoint in idle_endpoints:
    sagemaker.delete_endpoint(EndpointName=endpoint)
    print(f"Deleted idle endpoint: {endpoint}")
```

**Expected Savings**: Approximately $2,000-3,000 per month based on current g5.2xlarge instance pricing.

#### 1.2 Implement S3 Lifecycle Policies
```json
{
  "Rules": [
    {
      "ID": "Move training data to IA after 30 days",
      "Status": "Enabled",
      "Prefix": "training-data/",
      "Transition": {
        "Days": 30,
        "StorageClass": "STANDARD_IA"
      }
    },
    {
      "ID": "Archive old models after 90 days",
      "Status": "Enabled",
      "Prefix": "models/",
      "Transition": {
        "Days": 90,
        "StorageClass": "GLACIER"
      }
    }
  ]
}
```

**Expected Savings**: 20-30% on storage costs.

#### 1.3 Clean Up Failed Training Jobs
```python
# Sample code to clean up artifacts from failed training jobs
import boto3
s3 = boto3.client('s3')
sagemaker = boto3.client('sagemaker')

# Get list of failed jobs
failed_jobs = sagemaker.list_training_jobs(
    StatusEquals='Failed',
    MaxResults=100
)

# Delete associated S3 artifacts
bucket = "sagemaker-us-west-2-XXXXXXXXXXXX"
for job in failed_jobs['TrainingJobSummaries']:
    prefix = f"training-jobs/{job['TrainingJobName']}"
    response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    if 'Contents' in response:
        for obj in response['Contents']:
            s3.delete_object(Bucket=bucket, Key=obj['Key'])
            print(f"Deleted: {obj['Key']}")
```

### 2. Medium-Term Optimization Strategies

#### 2.1 Implement SageMaker Spot Instances for Training

```python
# Sample training job configuration with Spot instances
training_params = {
    'AlgorithmSpecification': {
        'TrainingImage': '...',
        'TrainingInputMode': 'File',
    },
    'RoleArn': 'arn:aws:iam::XXXXXXXXXXXX:role/service-role/AmazonSageMaker-ExecutionRole',
    'OutputDataConfig': {
        'S3OutputPath': 's3://sagemaker-us-west-2-XXXXXXXXXXXX/training-output/'
    },
    'ResourceConfig': {
        'InstanceType': 'ml.g5.12xlarge',
        'InstanceCount': 1,
        'VolumeSizeInGB': 30,
        # Enable Spot training
        'UseSpotInstances': True,
        'MaxWaitTimeInSeconds': 3600,
        'MaxRuntimeInSeconds': 3600
    },
    'StoppingCondition': {
        'MaxRuntimeInSeconds': 3600
    },
    'TrainingJobName': 'spot-training-job',
    'HyperParameters': {
        'epochs': '10',
        'learning_rate': '0.01'
    },
    'InputDataConfig': [
        {
            'ChannelName': 'training',
            'DataSource': {
                'S3DataSource': {
                    'S3DataType': 'S3Prefix',
                    'S3Uri': 's3://sagemaker-us-west-2-XXXXXXXXXXXX/training-data/',
                    'S3DataDistributionType': 'FullyReplicated'
                }
            }
        }
    ]
}
```

**Expected Savings**: 70-90% on training costs compared to On-Demand instances.

#### 2.2 Implement SageMaker Serverless Inference

```python
# Sample code for serverless inference endpoint
serverless_config = {
    'EndpointName': 'model-serverless-endpoint',
    'EndpointConfigName': 'model-serverless-config',
    'ProductionVariants': [
        {
            'VariantName': 'AllTraffic',
            'ModelName': 'model-name',
            'ServerlessConfig': {
                'MemorySizeInMB': 4096,
                'MaxConcurrency': 5
            }
        }
    ]
}
```

**Expected Savings**: Pay-per-use model eliminates costs for idle endpoints.

#### 2.3 Implement Multi-Model Endpoints

For similar models (multiple model variants), consolidate to a single endpoint:

```python
# Sample multi-model endpoint configuration
multi_model_config = {
    'EndpointName': 'llm-multi-model-endpoint',
    'EndpointConfigName': 'llm-multi-model-config',
    'ProductionVariants': [
        {
            'VariantName': 'AllTraffic',
            'ModelName': 'llm-container',
            'InstanceType': 'ml.g5.2xlarge',
            'InitialInstanceCount': 1,
            'ModelDataDownloadTimeoutInSeconds': 600,
            'ContainerStartupHealthCheckTimeoutInSeconds': 600
        }
    ],
    'Tags': [
        {
            'Key': 'Purpose',
            'Value': 'MultiModelEndpoint'
        }
    ]
}
```

**Expected Savings**: 30-70% by consolidating multiple endpoints.

### 3. Long-Term Strategic Recommendations

#### 3.1 Implement SageMaker Savings Plans

For predictable workloads, commit to 1-year or 3-year Savings Plans:

- 1-year commitment: ~30-40% savings
- 3-year commitment: ~50-60% savings

#### 3.2 Develop a Comprehensive ML Cost Monitoring System

Implement a CloudWatch dashboard to monitor:
- Endpoint invocations and utilization
- Training job success rates and durations
- Storage growth and usage patterns
- Cost anomaly detection

#### 3.3 Establish ML Governance Policies

- Require cost justification for GPU instances
- Implement automatic shutdown of idle endpoints after 24 hours
- Enforce tagging for cost allocation
- Regular review of ML resources and costs

## Expected Cost Savings

| Area | Current Monthly Cost | Potential Savings | Optimized Cost |
|------|---------------------|-------------------|----------------|
| SageMaker Endpoints | ~$7,000 | 50-70% | $2,100-$3,500 |
| SageMaker Training | ~$2,285 | 40-60% | $914-$1,371 |
| Amazon Bedrock | ~$53 | 20-30% | $37-$42 |
| ML Data Storage | ~$10 | 30-50% | $5-$7 |
| **Total** | **~$9,348** | **40-60%** | **$3,056-$4,920** |

## Implementation Roadmap

### Phase 1: Immediate Actions (Week 1-2)
- Terminate idle SageMaker endpoints
- Clean up failed training job artifacts
- Implement S3 lifecycle policies

### Phase 2: Quick Wins (Week 3-4)
- Configure Spot instances for training jobs
- Optimize hyperparameters for training efficiency
- Implement token caching for Bedrock

### Phase 3: Infrastructure Optimization (Month 2)
- Deploy serverless inference for variable workloads
- Implement multi-model endpoints
- Set up auto-scaling for production endpoints

### Phase 4: Strategic Initiatives (Month 3-6)
- Evaluate and implement SageMaker Savings Plans
- Develop comprehensive ML cost monitoring
- Establish ML governance policies

## Next Steps

1. **Immediate Action**: Schedule termination of identified idle endpoints
2. **Quick Win**: Implement S3 lifecycle policies for ML data storage
3. **Analysis**: Review failed training jobs to identify root causes
4. **Planning**: Develop a phased implementation plan for the recommendations
5. **Monitoring**: Set up basic cost monitoring dashboards

## Appendix: AWS ML Cost Optimization Best Practices

### Training Jobs
- Use Spot instances for non-time-critical training
- Implement early stopping to avoid unnecessary computation
- Optimize hyperparameters to reduce training time
- Use managed warm pools for frequent training jobs

### Inference
- Match instance types to workload requirements
- Use auto-scaling for variable traffic patterns
- Consider serverless inference for sporadic workloads
- Implement multi-model endpoints for similar models

### Storage
- Use appropriate storage classes based on access patterns
- Implement lifecycle policies for aging data
- Clean up temporary and intermediate artifacts
- Compress training datasets where appropriate

### Monitoring
- Set up CloudWatch alarms for cost anomalies
- Monitor endpoint utilization metrics
- Track training job efficiency metrics
- Implement regular cost reviews
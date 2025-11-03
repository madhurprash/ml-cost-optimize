# AWS ML Workload Cost Optimization Agent

A CLI-powered deep research agent that analyzes machine learning workload costs in your AWS account and provides optimization recommendations. Built with Claude 3.7 Sonnet (via Amazon Bedrock or OpenAI), the agent specializes in SageMaker training jobs, inference endpoints, Amazon Bedrock usage, and ML data storage optimization. It uses multiple tools including ML service monitoring, CloudWatch analysis, cost reporting, and internet search to perform comprehensive ML cost analysis across AWS accounts.

## Prerequisites

- Python 3.12+
- AWS credentials configured with appropriate permissions (via AWS CLI or environment variables)
- Access to either:
  - Amazon Bedrock with Claude 3.7 Sonnet model (`us.anthropic.claude-3-7-sonnet-20250219-v1:0`), OR
  - OpenAI API with GPT-4 access
- [Tavily API](https://tavily.com/) account and API key for internet search
- [LangSmith](https://smith.langchain.com/) account for tracing (optional but recommended)

### Required AWS Permissions

The agent requires the following AWS permissions:
- **Amazon Bedrock**:
  - `bedrock:InvokeModel` - To run Claude models
  - `bedrock:GetModelInvocationLoggingConfiguration` - To analyze Bedrock usage
- **SageMaker**:
  - `sagemaker:ListTrainingJobs` - To list training jobs
  - `sagemaker:DescribeTrainingJob` - To get training job details
  - `sagemaker:ListEndpoints` - To list inference endpoints
  - `sagemaker:DescribeEndpoint` - To get endpoint details
  - `sagemaker:DescribeEndpointConfig` - To analyze endpoint configurations
- **CloudWatch & Logs**:
  - `cloudwatch:ListDashboards` - To list CloudWatch dashboards
  - `cloudwatch:GetDashboard` - To retrieve dashboard details
  - `cloudwatch:DescribeAlarms` - To check alarm status
  - `cloudwatch:GetMetricStatistics` - To retrieve metrics for ML services
  - `logs:DescribeLogGroups` - To list log groups
  - `logs:FilterLogEvents` - To retrieve and analyze logs
- **Cost Explorer**:
  - `ce:GetCostAndUsage` - To analyze ML service costs
- **S3**:
  - `s3:ListBucket` - To identify ML data storage
  - `s3:GetBucketLocation` - To analyze storage locations
- **IAM**:
  - `sts:AssumeRole` - For cross-account access (if needed)

## Features

- **Deep Research Capabilities**: Uses the [deepagents](https://github.com/anthropics/deepagents) library for multi-step reasoning
- **ML-Specific Cost Analysis**:
  - SageMaker training job cost analysis and optimization
  - Inference endpoint utilization and cost monitoring
  - Amazon Bedrock token usage and pricing analysis
  - ML data storage (S3) cost optimization
- **CloudWatch Integration**: Monitors ML service metrics, logs, alarms, and dashboards
- **Cross-Account Support**: Analyze ML costs across multiple AWS accounts
- **Internet Search**: Research latest ML optimization techniques and AWS best practices
- **Cost Recommendations**: Get actionable recommendations for ML workload optimization
- **Extended Timeout**: Configured with 200-minute timeout for long-running deep agent operations

## Installation

### Option 1: Install from Source

For development or to get the latest features:

```bash
# Clone the repository
git clone https://github.com/madhurprash/ml-cost-optimize.git
cd cost-analysis

# Install in editable mode
pip install -e .

# Or install with development dependencies
pip install -e ".[dev]"
```

### Option 2: Using uv (Alternative Package Manager)

```bash
# Install dependencies
uv sync

# The package will be available in the virtual environment
```

## Configuration

### Option 1: Environment Variables (Recommended)

Set up environment variables for your credentials:

```bash
# Required: Tavily API Key for internet search
export TAVILY_API_KEY=your_tavily_api_key_here

# AWS credentials (if not using AWS CLI default profile)
export AWS_PROFILE=your_profile_name
export AWS_REGION=us-east-1

# Optional: For OpenAI provider
export OPENAI_API_KEY=your_openai_api_key

# Optional: LangSmith for tracing
export LANGSMITH_API_KEY=your_langsmith_api_key
export LANGSMITH_TRACING=true
export LANGCHAIN_PROJECT=aws-cost-analysis
```

### Option 2: Using .env File

Create a `.env` file in the project root:

```bash
# Required: Tavily API Key for internet search
TAVILY_API_KEY=your_tavily_api_key_here

# Optional: OpenAI API key (if using OpenAI provider)
OPENAI_API_KEY=your_openai_api_key

# Optional: LangSmith for tracing
LANGSMITH_API_KEY=your_langsmith_api_key
LANGSMITH_TRACING=true
LANGCHAIN_PROJECT=aws-cost-analysis

# AWS credentials (if not using AWS CLI default profile)
AWS_PROFILE=your_profile_name
AWS_REGION=us-east-1
```

### Model Configuration

Edit `config.yaml` to customize the agent's model and inference parameters:

```yaml
model_information:
  deep_agent_model_info:
    model_id: us.anthropic.claude-3-7-sonnet-20250219-v1:0
    system_prompt_fpath: system_prompts/deep_agent_system_prompt.txt
    inference_parameters:
      temperature: 0.1
      max_tokens: 2048
      top_p: 0.92
      caching: true
```

## Usage

### Basic Usage

After installation, you can run the CLI tool from anywhere:

```bash
# Set your API key
export TAVILY_API_KEY=your_key_here

# Run a cost analysis query
ml-cost-optimize --query "Analyze my SageMaker costs and suggest optimizations"
```

### CLI Options

```bash
ml-cost-optimize --help
```

#### Required Arguments

- `--query`: The cost analysis query to run

#### API Keys and Credentials

- `--tavily-api-key`: Tavily API key (or set `TAVILY_API_KEY` env var)
- `--openai-api-key`: OpenAI API key if using OpenAI provider (or set `OPENAI_API_KEY` env var)
- `--aws-profile`: AWS profile name (or set `AWS_PROFILE` env var)
- `--aws-region`: AWS region (or set `AWS_REGION` env var)

#### Model Configuration

- `--provider`: Choose `bedrock` or `openai` (defaults to config.yaml setting)
- `--config`: Path to configuration file (default: `config.yaml`)

#### Output and Behavior

- `--output-file`: Save agent response to file
- `--debug`: Enable debug logging
- `--max-retries`: Maximum retries for tool errors (default: 3)
- `--root-dir`: Root directory for filesystem backend (default: current directory)

#### LangSmith Tracing

- `--langsmith-api-key`: LangSmith API key (or set `LANGSMITH_API_KEY` env var)
- `--langsmith-project`: LangSmith project name (or set `LANGCHAIN_PROJECT` env var)

### Example Usage

#### Basic Analysis with Environment Variables

```bash
# Set credentials
export TAVILY_API_KEY=tvly-xxx
export AWS_PROFILE=my-profile

# Run analysis
ml-cost-optimize --query "Analyze my ML workload costs and provide recommendations"
```

#### Using CLI Arguments

```bash
ml-cost-optimize \
  --tavily-api-key tvly-xxx \
  --aws-profile my-profile \
  --query "Review SageMaker endpoint costs and suggest optimizations"
```

#### Using OpenAI Instead of Bedrock

```bash
export TAVILY_API_KEY=tvly-xxx
export OPENAI_API_KEY=sk-xxx

ml-cost-optimize \
  --provider openai \
  --query "Analyze Amazon Bedrock usage and costs"
```

#### Save Output to File

```bash
ml-cost-optimize \
  --query "Comprehensive ML cost analysis" \
  --output-file analysis_report.json
```


### Example Queries

#### Comprehensive ML Workload Analysis
```bash
ml-cost-optimize --query "Analyze machine learning workload costs in my AWS account and create a comprehensive optimization report. Focus on SageMaker training jobs, endpoints, Bedrock usage, and ML data storage."
```

#### SageMaker Training Optimization
```bash
ml-cost-optimize --query "Analyze my SageMaker training jobs from the last 30 days. Identify opportunities to use Spot instances and optimize instance types."
```

#### Endpoint Cost Reduction
```bash
ml-cost-optimize --query "Review all active SageMaker endpoints, check their utilization, and recommend which endpoints should use auto-scaling or be shut down."
```

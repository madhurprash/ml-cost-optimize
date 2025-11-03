import os
import json
import logging
import boto3
from botocore.config import Config
from ml_cost_analysis.utils import *
from ml_cost_analysis.constants import *
from typing import Literal, Optional
from dotenv import load_dotenv
from tavily import TavilyClient
from langsmith import traceable
from langchain_aws import ChatBedrock
from langchain_openai import ChatOpenAI
from deepagents import create_deep_agent
from deepagents.backends import StateBackend, FilesystemBackend

# load the environment variables from the .env file
load_dotenv()

# Import ML and monitoring tools
from ml_cost_analysis.tools import (
    # ML-specific tools
    list_sagemaker_training_jobs,
    get_training_job_details,
    list_sagemaker_endpoints,
    get_endpoint_details,
    analyze_bedrock_usage,
    get_ml_cost_recommendations,
    analyze_ml_data_storage,
    # CloudWatch monitoring tools
    list_cloudwatch_dashboards,
    get_dashboard_summary,
    list_log_groups,
    fetch_cloudwatch_logs_for_service,
    analyze_log_group,
    get_cloudwatch_alarms_for_service,
    setup_cross_account_access,
)

# set a logger
logger = logging.getLogger(__name__)


def create_agent(
    config_file: str = "config.yaml",
    provider: Optional[str] = None,
    root_dir: Optional[str] = None,
):
    """Create and configure the deep research agent.

    Args:
        config_file: Path to configuration YAML file
        provider: LLM provider to use ('bedrock' or 'openai'). If None, uses config file setting
        root_dir: Root directory for filesystem backend. If None, uses current directory

    Returns:
        Configured deep agent instance
    """
    # load the configuration file
    config_data = load_config(config_file)
    logger.info(f"Loaded configuration from {config_file}")

    # initialize the Tavily client
    tavily_api_key = os.getenv("TAVILY_APY_KEY")
    if not tavily_api_key:
        raise ValueError(
            "TAVILY_APY_KEY environment variable not set. "
            "Please set it with: export TAVILY_API_KEY='your-api-key' or use --tavily-api-key"
        )
    tavily_client = TavilyClient(api_key=tavily_api_key)
    logger.info("Initialized Tavily client for internet search")

    def internet_search(
        query: str,
        max_results: int = 5,
        topic: Literal["general", "news", "finance"] = "general",
        include_raw_content: bool = False,
    ):
        """Run a web search"""
        return tavily_client.search(
            query,
            max_results=max_results,
            include_raw_content=include_raw_content,
            topic=topic,
        )

    # Get provider from config or parameter
    if provider is None:
        provider = config_data['model_information']["deep_agent_model_info"].get("provider", "bedrock")
    logger.info(f"Using provider: {provider}")

    # Load provider-specific configuration
    if provider == "openai":
        agent_config = config_data['model_information']["deep_agent_model_info"]["openai"]
        deep_agent_prompt_path = agent_config["system_prompt_fpath"]
        deep_agent_system_prompt = load_system_prompt(deep_agent_prompt_path)
        logger.debug(f"Loaded system prompt from {deep_agent_prompt_path}")

        # Get OpenAI API key from environment
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable not set. "
                "Please set it with: export OPENAI_API_KEY='your-api-key' or use --openai-api-key"
            )

        # Initialize OpenAI model
        deep_agent_model = ChatOpenAI(
            model=agent_config["model_id"],
            temperature=agent_config["inference_parameters"]["temperature"],
            max_tokens=agent_config["inference_parameters"]["max_tokens"],
            top_p=agent_config["inference_parameters"]["top_p"],
            openai_api_key=openai_api_key,
            request_timeout=12000,  # 200 minutes - accommodates long-running deep agent operations
        )
        logger.info(f"Initialized OpenAI model: {agent_config['model_id']}")

    elif provider == "bedrock":
        agent_config = config_data['model_information']["deep_agent_model_info"]["bedrock"]
        deep_agent_prompt_path = agent_config["system_prompt_fpath"]
        deep_agent_system_prompt = load_system_prompt(deep_agent_prompt_path)
        logger.debug(f"Loaded system prompt from {deep_agent_prompt_path}")

        # Create a boto3 config with extended timeout for long-running deep agent queries
        # Default timeout of 60s is insufficient for deep agents that may run 10-15+ minutes
        bedrock_config = Config(
            read_timeout=12000,  # 200 minutes - accommodates long-running deep agent operations
            connect_timeout=60,
            retries={
                'max_attempts': 3,
                'mode': 'adaptive'
            }
        )

        # Create a boto3 client with custom timeout configuration
        bedrock_runtime_client = boto3.client(
            service_name='bedrock-runtime',
            region_name=boto3.session.Session().region_name,
            config=bedrock_config
        )

        deep_agent_model = ChatBedrock(
            client=bedrock_runtime_client,  # Use custom client with extended timeout
            model=agent_config["model_id"],
            temperature=agent_config["inference_parameters"]["temperature"],
            max_tokens=agent_config["inference_parameters"]["max_tokens"],
            top_p=agent_config["inference_parameters"]["top_p"],
        )
        logger.info(f"Initialized Amazon Bedrock model: {agent_config['model_id']}")

    else:
        raise ValueError(f"Unsupported provider: {provider}. Must be 'openai' or 'bedrock'")

    # Use provided root_dir or default to current directory
    if root_dir is None:
        root_dir = os.getcwd()

    deep_agent = create_deep_agent(
        tools=[
            # Internet search
            internet_search,
            # ML-specific tools
            list_sagemaker_training_jobs,
            get_training_job_details,
            list_sagemaker_endpoints,
            get_endpoint_details,
            analyze_bedrock_usage,
            get_ml_cost_recommendations,
            analyze_ml_data_storage,
            # CloudWatch monitoring tools
            list_cloudwatch_dashboards,
            get_dashboard_summary,
            list_log_groups,
            fetch_cloudwatch_logs_for_service,
            analyze_log_group,
            get_cloudwatch_alarms_for_service,
            setup_cross_account_access,
        ],
        model=deep_agent_model,
        system_prompt=deep_agent_system_prompt,
        backend=FilesystemBackend(root_dir=root_dir),
    )

    logger.info("Deep research agent created successfully")
    return deep_agent


@traceable
def run_deep_agent_query(
    agent,
    query: str,
    max_retries: int = 3,
) -> dict:
    """Run a query through the deep research agent with retry logic.

    Args:
        agent: The deep agent instance to use
        query: The user query to process
        max_retries: Maximum number of retry attempts for tool_use errors

    Returns:
        The agent's response as a dictionary
    """
    logger.info(f"Running deep agent query: {query}")

    for attempt in range(max_retries):
        try:
            result = agent.invoke({"messages": [{"role": "user", "content": query}]})
            logger.info("Deep agent query completed")
            return result
        except Exception as e:
            error_msg = str(e)

            # Check if it's a retryable error (tool errors or validation errors)
            is_retryable = (
                "tool_use" in error_msg and "tool_result" in error_msg
            ) or (
                "validation error" in error_msg.lower() and "write_file" in error_msg
            )

            if is_retryable:
                logger.warning(
                    f"Tool error on attempt {attempt + 1}/{max_retries}: {error_msg}"
                )

                if attempt < max_retries - 1:
                    logger.info(f"Retrying... (attempt {attempt + 2}/{max_retries})")
                    continue
                else:
                    logger.error(
                        "Max retries reached. The agent is having trouble with tool usage. "
                        "Consider simplifying the query or reducing the number of tools."
                    )
                    raise
            else:
                # Re-raise other exceptions immediately
                raise

    # This should never be reached, but just in case
    raise RuntimeError("Unexpected error in retry logic")


def main():
    """Main function to run the deep research agent.

    This is a standalone version for backward compatibility.
    For CLI usage, use the cli.py module instead.
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
    )

    # Create the agent
    agent = create_agent()

    # Example query - you can modify this
    query = """Analyze machine learning workload costs in my AWS account and create a comprehensive
    optimization report. Focus on:
    1. SageMaker training jobs and their costs
    2. SageMaker endpoint utilization and recommendations
    3. Amazon Bedrock usage patterns
    4. ML data storage optimization opportunities
    5. Specific cost reduction strategies for ML workloads"""

    logger.info("Starting ML workload optimization analysis...")
    result = run_deep_agent_query(agent, query)

    # Print the agent's response
    print("\n" + "="*80)
    print("AGENT RESPONSE")
    print("="*80)
    print(json.dumps(result, indent=2, default=str))
    print("="*80)


if __name__ == "__main__":
    main()
import os
import json
import boto3
from botocore.config import Config
from utils import *
from constants import *
from typing import Literal
from dotenv import load_dotenv
from tavily import TavilyClient
from langsmith import traceable
from langchain_aws import ChatBedrock
from deepagents import create_deep_agent

# load the environment variables from the .env file
load_dotenv()

# Import CloudWatch monitoring tools
from tools import (
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
logger.setLevel(logging.INFO)

# load the configuration file
config_data = load_config(CONFIG_FILE_PATH)
print(f"Loaded the configuration file for the deep agents: {json.dumps(config_data, indent=2)}")

# initialize the Tavily client
tavily_api_key = os.getenv("TAVILY_APY_KEY")
tavily_client = TavilyClient(api_key=tavily_api_key)
print("Initialized the Tavily client for internet search tool.")

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

# Next, we will create the deep research agent
print(f"Loading the system prompt for the deep agent from the config file path...")
deep_agent_prompt_path = config_data['model_information']["deep_agent_model_info"]["system_prompt_fpath"]
deep_agent_system_prompt = load_system_prompt(deep_agent_prompt_path)
print(f"Loaded the deep agent system prompt: {deep_agent_system_prompt}")

# load the deep agent model configuration
agent_config = config_data['model_information']["deep_agent_model_info"]

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
print(f"Initialized the deep agent model with 200-minute timeout: {deep_agent_model}")

deep_agent = create_deep_agent(
    tools=[internet_search, 
           list_cloudwatch_dashboards,
           get_dashboard_summary,
           list_log_groups,
           fetch_cloudwatch_logs_for_service,
           analyze_log_group,
           get_cloudwatch_alarms_for_service,
           setup_cross_account_access],
    model = deep_agent_model,
    system_prompt=deep_agent_system_prompt,
)

print(f"Created the deep research agent with the specified tools and model: {deep_agent}")


@traceable
def run_deep_agent_query(query: str) -> dict:
    """Run a query through the deep research agent.

    Args:
        query: The user query to process

    Returns:
        The agent's response as a dictionary
    """
    logger.info(f"Running deep agent query: {query}")
    result = deep_agent.invoke({"messages": [{"role": "user", "content": query}]})
    logger.info("Deep agent query completed")
    return result


def main():
    """Main function to run the deep research agent."""
    # Example query - you can modify this or add command-line argument parsing
    query = "Analyze cost expenditure across my account and create a thorough report on optimization techniques."

    logger.info("Starting deep research agent...")
    result = run_deep_agent_query(query)

    # Print the agent's response
    print("\n" + "="*80)
    print("AGENT RESPONSE")
    print("="*80)
    print(json.dumps(result, indent=2, default=str))
    print("="*80)


if __name__ == "__main__":
    main()
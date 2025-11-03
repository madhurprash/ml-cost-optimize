#!/usr/bin/env python3
"""Command-line interface for AWS ML Cost Analysis Agent."""

import argparse
import logging
import os
import sys
import time
from typing import Optional


def _get_env_value(
    cli_value: Optional[str],
    env_var: str,
    required: bool = True,
) -> Optional[str]:
    """Get configuration value from CLI or environment variable.

    Args:
        cli_value: Value provided via CLI argument
        env_var: Environment variable name to check
        required: Whether the value is required

    Returns:
        The configuration value

    Raises:
        ValueError: If required value is not provided
    """
    if cli_value:
        return cli_value

    env_value = os.getenv(env_var)
    if env_value:
        return env_value

    if required:
        raise ValueError(
            f"Value must be provided via CLI argument or {env_var} environment variable"
        )

    return None


def _setup_logging(debug: bool = False) -> None:
    """Configure logging with basicConfig.

    Args:
        debug: Whether to enable debug logging
    """
    level = logging.DEBUG if debug else logging.INFO

    logging.basicConfig(
        level=level,
        format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
    )


def _parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed command-line arguments
    """
    parser = argparse.ArgumentParser(
        description="Analyze AWS ML workload costs and provide optimization recommendations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
    # Using environment variables (recommended)
    export TAVILY_API_KEY=your_tavily_key
    export AWS_PROFILE=your_profile
    ml-cost-analysis --query "Analyze my SageMaker costs"

    # Using CLI arguments
    ml-cost-analysis --tavily-api-key tvly-xxx --aws-profile my-profile --query "..."

    # Using OpenAI instead of Bedrock
    export OPENAI_API_KEY=sk-xxx
    ml-cost-analysis --provider openai --query "..."

    # Enable debug logging
    ml-cost-analysis --debug --query "..."

    # Use custom config file
    ml-cost-analysis --config custom-config.yaml --query "..."
""",
    )

    # Required arguments
    parser.add_argument(
        "--query",
        type=str,
        help="Cost analysis query to run",
    )

    # API keys and credentials
    parser.add_argument(
        "--tavily-api-key",
        type=str,
        help="Tavily API key for internet search (or set TAVILY_API_KEY env var)",
    )

    parser.add_argument(
        "--openai-api-key",
        type=str,
        help="OpenAI API key (required if using OpenAI provider, or set OPENAI_API_KEY env var)",
    )

    # AWS configuration
    parser.add_argument(
        "--aws-profile",
        type=str,
        help="AWS profile name to use (or set AWS_PROFILE env var)",
    )

    parser.add_argument(
        "--aws-region",
        type=str,
        help="AWS region to use (or set AWS_REGION env var)",
    )

    # Model configuration
    parser.add_argument(
        "--provider",
        type=str,
        choices=["bedrock", "openai"],
        help="LLM provider to use (bedrock or openai). Defaults to config.yaml setting",
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file (default: config.yaml)",
    )

    # LangSmith tracing
    parser.add_argument(
        "--langsmith-api-key",
        type=str,
        help="LangSmith API key for tracing (optional, or set LANGSMITH_API_KEY env var)",
    )

    parser.add_argument(
        "--langsmith-project",
        type=str,
        help="LangSmith project name (optional, or set LANGCHAIN_PROJECT env var)",
    )

    # Output and behavior
    parser.add_argument(
        "--output-file",
        type=str,
        help="Save agent response to file (optional)",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )

    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum number of retries for tool errors (default: 3)",
    )

    parser.add_argument(
        "--root-dir",
        type=str,
        default=os.getcwd(),
        help="Root directory for filesystem backend (default: current directory)",
    )

    return parser.parse_args()


def main() -> None:
    """Main CLI entry point."""
    args = _parse_arguments()

    # Setup logging
    _setup_logging(args.debug)
    logger = logging.getLogger(__name__)

    # Validate query is provided
    if not args.query:
        logger.error("Error: --query argument is required")
        sys.exit(1)

    # Set up environment variables from CLI args
    try:
        # Tavily API key (required)
        tavily_key = _get_env_value(
            args.tavily_api_key,
            "TAVILY_API_KEY",
            required=True,
        )
        os.environ["TAVILY_APY_KEY"] = tavily_key

        # AWS configuration (optional)
        if args.aws_profile:
            os.environ["AWS_PROFILE"] = args.aws_profile

        if args.aws_region:
            os.environ["AWS_REGION"] = args.aws_region

        # LangSmith configuration (optional)
        if args.langsmith_api_key:
            os.environ["LANGSMITH_API_KEY"] = args.langsmith_api_key
            os.environ["LANGSMITH_TRACING"] = "true"

        if args.langsmith_project:
            os.environ["LANGCHAIN_PROJECT"] = args.langsmith_project

    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(1)

    # Now import the agent module (after environment setup)
    try:
        from long_running_deep_agent import (
            create_agent,
            run_deep_agent_query,
        )
    except ImportError as e:
        logger.error(f"Failed to import agent module: {e}")
        logger.error("Make sure all dependencies are installed: uv sync")
        sys.exit(1)

    # Create the agent
    try:
        logger.info("Initializing AWS ML Cost Analysis Agent...")

        # Create agent with configuration
        agent_config = {
            "config_file": args.config,
            "provider": args.provider,
            "root_dir": args.root_dir,
        }

        # Check if OpenAI is being used and validate API key
        if args.provider == "openai":
            openai_key = _get_env_value(
                args.openai_api_key,
                "OPENAI_API_KEY",
                required=True,
            )
            os.environ["OPENAI_API_KEY"] = openai_key

        agent = create_agent(**agent_config)
        logger.info("Agent initialized successfully")

    except Exception as e:
        logger.error(f"Failed to initialize agent: {e}")
        sys.exit(1)

    # Run the query
    try:
        logger.info(f"Running query: {args.query}")
        start_time = time.time()

        result = run_deep_agent_query(
            agent,
            args.query,
            max_retries=args.max_retries,
        )

        elapsed_time = time.time() - start_time
        minutes = int(elapsed_time // 60)
        seconds = elapsed_time % 60

        if minutes > 0:
            logger.info(f"Query completed in {minutes} minutes and {seconds:.1f} seconds")
        else:
            logger.info(f"Query completed in {seconds:.1f} seconds")

        # Output results
        import json

        output = json.dumps(result, indent=2, default=str)

        print("\n" + "="*80)
        print("AGENT RESPONSE")
        print("="*80)
        print(output)
        print("="*80)

        # Save to file if requested
        if args.output_file:
            with open(args.output_file, "w") as f:
                f.write(output)
            logger.info(f"Response saved to {args.output_file}")

    except Exception as e:
        logger.error(f"Query failed: {e}")
        if args.debug:
            logger.exception("Full traceback:")
        sys.exit(1)


if __name__ == "__main__":
    main()

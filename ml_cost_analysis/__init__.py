"""AWS ML Workload Cost Optimization Agent.

A CLI-powered deep research agent that analyzes machine learning workload
costs in your AWS account and provides optimization recommendations.
"""

__version__ = "0.1.0"
__author__ = "Your Name"

from ml_cost_analysis.agent import (
    create_agent,
    run_deep_agent_query,
)

__all__ = [
    "create_agent",
    "run_deep_agent_query",
]

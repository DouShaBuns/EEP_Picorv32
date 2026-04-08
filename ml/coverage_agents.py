"""
coverage_agents.py
Shared coverage-agent factory and persistence helpers.
"""

from dqn_agent import DQNAgent
from tree_agents import DecisionTreeCoverageAgent, RandomForestCoverageAgent

COVERAGE_AGENT_CHOICES = ("dqn", "rf", "dt")


def build_coverage_agent(agent_type: str):
    if agent_type == "dqn":
        return DQNAgent()
    if agent_type == "rf":
        return RandomForestCoverageAgent()
    if agent_type == "dt":
        return DecisionTreeCoverageAgent()
    raise ValueError(f"Unsupported coverage agent: {agent_type}")


def coverage_agent_model_filename(agent_type: str) -> str:
    if agent_type == "dqn":
        return "dqn_model.pt"
    if agent_type == "rf":
        return "rf_model.pkl"
    if agent_type == "dt":
        return "dt_model.pkl"
    raise ValueError(f"Unsupported coverage agent: {agent_type}")

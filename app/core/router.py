from enum import Enum
from typing import Dict, Any, Optional
from app.utils.logger import logger

class AgentType(Enum):
    GENERAL_QA = "general_qa"
    TECHNICAL = "technical"
    CREATIVE = "creative"
    CODE = "code"
    MATH = "math"

class QueryRouter:
    def __init__(self):
        self.routing_rules = {
            AgentType.GENERAL_QA: self._is_general_qa,
            AgentType.TECHNICAL: self._is_technical,
            AgentType.CREATIVE: self._is_creative,
            AgentType.CODE: self._is_code,
            AgentType.MATH: self._is_math,
        }

    def route_query(self, query: str) -> AgentType:
        """Route a query to the most appropriate agent."""
        query_lower = query.lower()

        for agent_type, rule_func in self.routing_rules.items():
            if rule_func(query_lower):
                logger.info(f"Routed query to {agent_type.value}: {query[:50]}...")
                return agent_type

        # Default to general QA
        logger.info(f"Default routing to general_qa: {query[:50]}...")
        return AgentType.GENERAL_QA

    def _is_general_qa(self, query: str) -> bool:
        """Check if query is general question-answering."""
        general_keywords = [
            "what", "who", "when", "where", "why", "how",
            "explain", "describe", "tell me about"
        ]
        return any(keyword in query for keyword in general_keywords)

    def _is_technical(self, query: str) -> bool:
        """Check if query is technical."""
        technical_keywords = [
            "api", "database", "server", "network", "security",
            "algorithm", "framework", "library", "protocol",
            "architecture", "infrastructure", "deployment"
        ]
        return any(keyword in query for keyword in technical_keywords)

    def _is_creative(self, query: str) -> bool:
        """Check if query requires creative thinking."""
        creative_keywords = [
            "design", "create", "imagine", "brainstorm",
            "innovative", "creative", "story", "poem",
            "artwork", "music", "fiction"
        ]
        return any(keyword in query for keyword in creative_keywords)

    def _is_code(self, query: str) -> bool:
        """Check if query is about programming/code."""
        code_keywords = [
            "code", "program", "function", "class", "variable",
            "python", "javascript", "java", "c++", "sql",
            "debug", "error", "exception", "syntax"
        ]
        return any(keyword in query for keyword in code_keywords)

    def _is_math(self, query: str) -> bool:
        """Check if query is mathematical."""
        math_keywords = [
            "calculate", "compute", "solve", "equation",
            "formula", "theorem", "proof", "integral",
            "derivative", "matrix", "probability"
        ]
        math_symbols = ["=", "+", "-", "*", "/", "^", "√", "∑", "∫"]
        has_math_symbols = any(symbol in query for symbol in math_symbols)

        return any(keyword in query for keyword in math_keywords) or has_math_symbols

class AgentConfig:
    """Configuration for different agents."""

    @staticmethod
    def get_agent_config(agent_type: AgentType) -> Dict[str, Any]:
        """Get configuration for a specific agent type."""
        configs = {
            AgentType.GENERAL_QA: {
                "model": "llama3-8b-8192",
                "temperature": 0.2,
                "max_tokens": 1000,
                "system_prompt": "You are a helpful assistant. Answer questions based on the provided context."
            },
            AgentType.TECHNICAL: {
                "model": "llama3-70b-8192",
                "temperature": 0.1,
                "max_tokens": 1500,
                "system_prompt": "You are a technical expert. Provide detailed, accurate technical information."
            },
            AgentType.CREATIVE: {
                "model": "llama3-70b-8192",
                "temperature": 0.8,
                "max_tokens": 1200,
                "system_prompt": "You are a creative assistant. Be imaginative and help with creative tasks."
            },
            AgentType.CODE: {
                "model": "llama3-70b-8192",
                "temperature": 0.1,
                "max_tokens": 2000,
                "system_prompt": "You are a programming expert. Provide accurate code solutions and explanations."
            },
            AgentType.MATH: {
                "model": "llama3-70b-8192",
                "temperature": 0.0,
                "max_tokens": 1000,
                "system_prompt": "You are a mathematics expert. Solve problems step by step with clear explanations."
            }
        }
        return configs.get(agent_type, configs[AgentType.GENERAL_QA])
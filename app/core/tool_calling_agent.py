from typing import List, Dict, Any, Optional, Callable
from app.core.llm import generate_answer, client
from app.core.router import AgentType
from app.tools.health import health_check
from app.tools.ingest import ingest_documents
from app.tools.search import search_knowledge
from app.tools.answer import answer_question
from app.utils.logger import logger
import json
import re

class ToolCallingAgent:
    def __init__(self):
        self.available_tools = {
            "health_check": {
                "function": health_check,
                "description": "Check the health status of the RAG system",
                "parameters": {}
            },
            "ingest_documents": {
                "function": self._ingest_wrapper,
                "description": "Ingest documents into the vector store",
                "parameters": {
                    "documents": "List of strings representing documents to ingest"
                }
            },
            "search_knowledge": {
                "function": search_knowledge,
                "description": "Search the knowledge base for relevant documents",
                "parameters": {
                    "query": "Search query string",
                    "top_k": "Number of results to return (optional, default 5)"
                }
            },
            "answer_question": {
                "function": self._answer_wrapper,
                "description": "Answer a question using the RAG system",
                "parameters": {
                    "question": "Question to answer"
                }
            }
        }

    def _ingest_wrapper(self, documents: List[str]) -> Dict[str, Any]:
        """Wrapper for ingest_documents tool."""
        return ingest_documents(documents)

    def _answer_wrapper(self, question: str) -> Dict[str, Any]:
        """Wrapper for answer_question tool."""
        return answer_question(question)

    def execute_with_tools(self, user_query: str, max_iterations: int = 5) -> Dict[str, Any]:
        """Execute a query using tool calling capabilities."""
        messages = [
            {
                "role": "system",
                "content": self._get_system_prompt()
            },
            {
                "role": "user",
                "content": user_query
            }
        ]

        for iteration in range(max_iterations):
            logger.info(f"Tool-calling iteration {iteration + 1}")

            try:
                # Get available tools for this turn
                tools = self._get_available_tools()

                response = client.chat.completions.create(
                    model="llama3-70b-8192",  # Use more capable model for tool calling
                    messages=messages,
                    tools=tools,
                    tool_choice="auto",
                    temperature=0.1
                )

                message = response.choices[0].message

                # Check if tool calls were made
                if hasattr(message, 'tool_calls') and message.tool_calls:
                    # Execute tool calls
                    tool_results = self._execute_tool_calls(message.tool_calls)

                    # Add assistant message with tool calls
                    messages.append({
                        "role": "assistant",
                        "content": message.content,
                        "tool_calls": message.tool_calls
                    })

                    # Add tool results
                    for result in tool_results:
                        messages.append(result)

                else:
                    # Final answer
                    final_answer = message.content
                    logger.info("Tool-calling agent completed with final answer")
                    return {
                        "answer": final_answer,
                        "tool_calls_made": iteration,
                        "agent_type": "tool_calling"
                    }

            except Exception as e:
                logger.error(f"Tool-calling iteration {iteration + 1} failed: {e}")
                # Fallback to regular RAG
                fallback_result = answer_question(user_query)
                return {
                    "answer": fallback_result.get("answer", "Error occurred"),
                    "tool_calls_made": 0,
                    "agent_type": "fallback_rag",
                    "error": str(e)
                }

        # Max iterations reached
        logger.warning("Max iterations reached in tool-calling agent")
        return {
            "answer": "I apologize, but I was unable to complete your request within the allowed iterations.",
            "tool_calls_made": max_iterations,
            "agent_type": "tool_calling_max_iterations"
        }

    def _get_system_prompt(self) -> str:
        """Get the system prompt for tool calling."""
        return """You are an intelligent assistant with access to various RAG (Retrieval-Augmented Generation) tools.

You can use the following tools:
- health_check: Check system status
- ingest_documents: Add new documents to the knowledge base
- search_knowledge: Search for information in the knowledge base
- answer_question: Get comprehensive answers using RAG

When responding:
1. Use tools when they would help answer the user's query
2. For complex queries, break them down and use multiple tools
3. Provide clear, helpful answers based on tool results
4. If you have enough information, provide a final answer

Always be helpful and accurate."""

    def _get_available_tools(self) -> List[Dict[str, Any]]:
        """Get the list of available tools in OpenAI function format."""
        tools = []
        for tool_name, tool_info in self.available_tools.items():
            tool_def = {
                "type": "function",
                "function": {
                    "name": tool_name,
                    "description": tool_info["description"],
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            }

            # Add parameters
            for param_name, param_desc in tool_info["parameters"].items():
                tool_def["function"]["parameters"]["properties"][param_name] = {
                    "type": "string",
                    "description": param_desc
                }
                tool_def["function"]["parameters"]["required"].append(param_name)

            tools.append(tool_def)

        return tools

    def _execute_tool_calls(self, tool_calls) -> List[Dict[str, Any]]:
        """Execute the tool calls and return results."""
        results = []

        for tool_call in tool_calls:
            try:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)

                logger.info(f"Executing tool: {function_name} with args: {function_args}")

                if function_name in self.available_tools:
                    tool_func = self.available_tools[function_name]["function"]
                    result = tool_func(**function_args)

                    results.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(result)
                    })
                else:
                    results.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps({"error": f"Unknown tool: {function_name}"})
                    })

            except Exception as e:
                logger.error(f"Tool execution failed: {e}")
                results.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps({"error": str(e)})
                })

        return results
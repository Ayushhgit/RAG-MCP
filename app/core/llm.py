from groq import Groq
from app.config import GROQ_API_KEY, LLM_MODEL
from app.core.router import AgentType, AgentConfig
from app.utils.logger import logger
import asyncio
from typing import AsyncGenerator, Optional

client = Groq(api_key=GROQ_API_KEY)

def generate_answer(context: str, question: str, agent_type: AgentType = AgentType.GENERAL_QA) -> str:
    """Generate answer using the appropriate agent configuration."""
    try:
        config = AgentConfig.get_agent_config(agent_type)

        # Use agent-specific model if available, otherwise fall back to default
        model = config.get("model", LLM_MODEL)
        temperature = config.get("temperature", 0.2)
        max_tokens = config.get("max_tokens", 1000)
        system_prompt = config.get("system_prompt", "Answer using the provided context only.")

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{question}"}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )

        logger.info(f"Generated answer using {agent_type.value} agent with model {model}")
        return response.choices[0].message.content

    except Exception as e:
        logger.error(f"Error generating answer with {agent_type.value}: {e}")
        # Fallback to default model
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "Answer using the provided context only."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{question}"}
            ],
            temperature=0.2
        )
        return response.choices[0].message.content

async def generate_answer_stream(context: str, question: str, agent_type: AgentType = AgentType.GENERAL_QA) -> AsyncGenerator[str, None]:
    """Generate streaming answer using the appropriate agent configuration."""
    try:
        config = AgentConfig.get_agent_config(agent_type)

        model = config.get("model", LLM_MODEL)
        temperature = config.get("temperature", 0.2)
        max_tokens = config.get("max_tokens", 1000)
        system_prompt = config.get("system_prompt", "Answer using the provided context only.")

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{question}"}
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True
        )

        logger.info(f"Streaming answer using {agent_type.value} agent with model {model}")

        full_response = ""
        for chunk in response:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                full_response += content
                yield content

        logger.info(f"Completed streaming response: {len(full_response)} chars")

    except Exception as e:
        logger.error(f"Error in streaming answer with {agent_type.value}: {e}")
        # Fallback to non-streaming
        fallback_response = generate_answer(context, question, agent_type)
        yield fallback_response

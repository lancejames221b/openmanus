"""
Claude 3.7 Extended Thinking Support Module

This module provides support for Claude 3.7 Sonnet's extended thinking capabilities,
allowing OpenManus to leverage Claude's advanced reasoning abilities.

Extended thinking allows Claude to perform step-by-step reasoning for complex tasks,
with a configurable "thinking budget" to control the amount of tokens used for reasoning.
"""

import json
from typing import Dict, List, Optional, Any, Union
import httpx
import socket
import asyncio

from app.config import config
from app.exceptions import TokenLimitExceeded
from app.logger import logger
from app.schema import Message
from app.llm import LLM
from openai.types.chat.chat_completion_message import ChatCompletionMessage


class Claude37Client(LLM):
    """Client for Claude 3.7 with Extended Thinking support."""

    def __init__(self, api_key: str = None, base_url: str = None, use_openrouter: bool = None):
        """Initialize the Claude 3.7 client.
        
        Args:
            api_key: API key for authentication (default: from config)
            base_url: Base URL for the API (default: from config)
            use_openrouter: Whether to use OpenRouter API (default: detect from config)
        """
        # Initialize the parent LLM class
        super().__init__("default")
        
        # Override with Claude-specific settings
        self.api_key = api_key or config.llm["default"].api_key
        
        # Make sure the base URL is properly formatted without trailing slash
        provided_url = base_url or config.llm["default"].base_url
        self.base_url = provided_url.rstrip("/")
        
        # Determine if we're using OpenRouter based on URL or explicit parameter
        self.use_openrouter = use_openrouter
        if self.use_openrouter is None:
            self.use_openrouter = "openrouter" in self.base_url.lower()
            
        # Set model name based on whether we're using OpenRouter
        if self.use_openrouter:
            self.model = "anthropic/claude-3.7-sonnet"
            # Make sure the base URL is correct for OpenRouter
            self.base_url = "https://openrouter.ai/api/v1"
        else:
            self.model = "claude-3.7-sonnet"
            # Ensure base URL points to messages endpoint for direct Anthropic
            if not self.base_url.endswith("/messages"):
                self.base_url = "https://api.anthropic.com/v1/messages"

        self.thinking_budget = config.llm["default"].thinking_budget if hasattr(config.llm["default"], "thinking_budget") else 32000
        self.extended_thinking = False
        self.max_tokens = config.llm["default"].max_tokens
        self.temperature = config.llm["default"].temperature
        
        logger.info(f"Initialized Claude 3.7 client with model: {self.model}, API: {'OpenRouter' if self.use_openrouter else 'Anthropic direct'}")

    async def ask(self, prompt: Union[str, List[Union[dict, Message]]], stream: bool = False) -> str:
        """
        Ask a question to Claude 3.7 (implements LLM.ask method for agent compatibility)
        
        Args:
            prompt: The prompt to send to Claude 3.7
            stream: Whether to stream the response
            
        Returns:
            str: The generated response
        """
        # Convert prompt to messages format if it's a string
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        else:
            messages = prompt
            
        # Use create_completion to get the response
        response = await self.create_completion(
            messages=messages,
            system_prompt="You are an AI assistant that helps users accomplish tasks.",
            thinking_budget=self.thinking_budget if self.extended_thinking else None,
            stream=stream
        )
        
        # Extract content from the response based on format
        if isinstance(response, dict):
            if "choices" in response and len(response["choices"]) > 0:
                # OpenRouter format
                choice = response["choices"][0]
                if "message" in choice and "content" in choice["message"]:
                    return choice["message"]["content"]
            elif "content" in response:
                # Direct Anthropic format
                return response["content"]
        
        # Handle string response or fallback
        return response if isinstance(response, str) else str(response)

    async def ask_tool(
        self,
        messages: List[Union[dict, Message]],
        system_msgs: Optional[List[Union[dict, Message]]] = None,
        timeout: int = 300,
        tools: Optional[List[dict]] = None,
        tool_choice: Any = "auto",
        stream: bool = False,
        temperature: Optional[float] = None,
        **kwargs,
    ) -> ChatCompletionMessage | None:
        """
        Ask LLM using functions/tools and return the response.
        
        This method delegates to the parent LLM class implementation to ensure
        tool use works correctly with the agent system.
        
        Args:
            messages: List of conversation messages
            system_msgs: Optional system messages to prepend
            timeout: Request timeout in seconds
            tools: List of tools to use
            tool_choice: Tool choice strategy
            stream: Whether to stream the response
            temperature: Sampling temperature for the response
            **kwargs: Additional completion arguments
            
        Returns:
            ChatCompletionMessage: The model's response
        """
        # For now, delegate to the parent LLM class implementation
        # This ensures tool use works while we develop Claude-specific tool handling
        return await super().ask_tool(
            messages=messages,
            system_msgs=system_msgs,
            timeout=timeout,
            tools=tools,
            tool_choice=tool_choice,
            stream=stream,
            temperature=temperature,
            **kwargs
        )

    async def create_completion(
        self, 
        messages: List[Dict[str, Any]], 
        system_prompt: Optional[str] = None,
        stream: bool = False,
        thinking_budget: Optional[int] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Create a completion using Claude 3.7 with extended thinking.
        
        Args:
            messages: List of conversation messages
            system_prompt: Optional system prompt
            stream: Whether to stream the response
            thinking_budget: Token budget for extended thinking
            max_tokens: Maximum tokens for the response
            temperature: Temperature for sampling
            
        Returns:
            Dict containing the completion response
        """
        # Create request payload
        payload = {
            "model": self.model
        }
        # Add default headers for OpenRouter
        headers = {
            "Content-Type": "application/json"
        }
        
        if self.use_openrouter:
            # OpenRouter uses OpenAI-compatible format
            payload.update({
                "messages": messages,
                "max_tokens": max_tokens or self.max_tokens,
                "temperature": temperature or self.temperature
            })
            
            # For Claude-3.7-sonnet with extended thinking, add system parameters
            if "claude-3.7" in self.model.lower() and self.extended_thinking:
                payload["stop_sequences"] = []
                payload["system"] = {
                    "extended_thinking": self.extended_thinking
                }
                if thinking_budget or self.thinking_budget:
                    payload["system"]["thinking_budget"] = thinking_budget or self.thinking_budget
                    
                logger.info(f"Added extended thinking with budget: {thinking_budget or self.thinking_budget}")
                
            # Add required HTTP headers for OpenRouter
            headers["Authorization"] = f"Bearer {self.api_key.strip()}"
            headers["HTTP-Referer"] = "https://openmanus.app"  # Site URL for request attribution
            headers["X-Title"] = "OpenManus"                   # App name for request attribution
            
            if system_prompt:
                # Handle system prompt for OpenRouter
                # For Claude with extended thinking via OpenRouter, add to system
                if "claude" in self.model.lower() and self.extended_thinking:
                    if "system" not in payload:
                        payload["system"] = {}
                    payload["system"]["prompt"] = system_prompt
                else:
                    # Standard OpenAI format - system message at beginning 
                    messages.insert(0, {"role": "system", "content": system_prompt})
        else:
            # Direct Anthropic API format
            payload.update({
                "messages": messages,
                "max_tokens": max_tokens or self.max_tokens,
                "temperature": temperature or self.temperature,
                "system": {
                "extended_thinking": self.extended_thinking,
                 "thinking_budget": thinking_budget or self.thinking_budget
                }
            })
            if system_prompt:
                payload["system"]["prompt"] = system_prompt
                    
        
        # Add API-specific authentication headers
        if self.use_openrouter:
            # Already added above
            pass
        else:
            headers["x-api-key"] = self.api_key.strip()
            headers["anthropic-version"] = "2023-06-01"
        
        try:
            # Use the chat completions endpoint for OpenRouter
            endpoint = "/chat/completions" if self.use_openrouter else ""  
            logger.info(f"Sending request to {self.base_url}{endpoint} with extended thinking (budget: {thinking_budget or self.thinking_budget})")
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}{endpoint}",
                    headers=headers,
                    json=payload,
                    timeout=120.0  # Extended timeout for complex reasoning
                )

                logger.debug(f"Request sent to {self.base_url}{endpoint}")
                logger.debug(f"Headers: {headers}")
                logger.debug(f"Payload: {payload}")
                
                if response.status_code != 200:
                    # Log detailed error information
                    try:
                        error_json = response.json()
                        logger.error(f"API error details: {error_json}")
                    except:
                        pass
                    error_text = response.text
                    logger.error(f"Claude API error: Status {response.status_code}: {error_text}")
                    # Try to use fallback if configured
                    if hasattr(config.llm, "fallback"):
                        logger.warning("Attempting to use fallback LLM configuration")
                        # Attempt to use the standard LLM with fallback config
                        from app.llm import LLM
                        fallback_llm = LLM("fallback")
                        # Convert messages to compatible format
                        formatted_messages = self._convert_to_openai_format(messages, system_prompt)
                        return await fallback_llm.ask(formatted_messages, stream=stream)
                    else:
                        response.raise_for_status()
                
                result = response.json()
                logger.debug(f"Response: {result}")
                
                logger.info("Successfully received response from Claude 3.7")
                return result
                
        except httpx.ConnectError as e:
            logger.error(f"Connection error: {e}")
            if "nodename nor servname provided" in str(e):
                logger.error("DNS resolution error. Check your network connection and URL validity.")
            
            # Try fallback
            if hasattr(config.llm, "fallback"):
                logger.warning("Connection error, using fallback LLM configuration")
                from app.llm import LLM
                fallback_llm = LLM("fallback")
                return await fallback_llm.ask(self._convert_to_openai_format(messages, system_prompt), stream=False)
        except Exception as e:  # Handle any other exceptions
            logger.error(f"Error in Claude 3.7 request: {e}")
            # Attempt fallback if available
            if hasattr(config.llm, "fallback"):
                logger.warning("Attempting to use fallback LLM configuration after exception")
                from app.llm import LLM
                fallback_llm = LLM("fallback")
                formatted_messages = self._convert_to_openai_format(messages, system_prompt)
                return await fallback_llm.ask(formatted_messages, stream=stream)
            raise
            
    def _convert_to_openai_format(self, messages: List[Dict[str, Any]], system_prompt: Optional[str] = None) -> List[Dict[str, Any]]:
        """Convert Claude message format to OpenAI format for fallback compatibility.
        
        Args:
            messages: Claude format messages
            system_prompt: Optional system prompt
            
        Returns:
            OpenAI-compatible message format
        """
        openai_messages = []
        
        # Add system message if provided
        if system_prompt:
            openai_messages.append({"role": "system", "content": system_prompt})
            
        # Convert message format
        for msg in messages:
            role = msg.get("role", "user") 
            content = msg.get("content", "")
            openai_messages.append({"role": role, "content": content})
            
        return openai_messages


async def test_claude_extended():
    """Test function for Claude 3.7 extended thinking capabilities."""
    client = Claude37Client()
    messages = [
        {"role": "user", "content": "Solve this step-by-step: If x^2 + 5x + 6 = 0, what are the values of x?"}
    ]
    
    response = await client.create_completion(
        messages=messages,
        system_prompt="You are a helpful assistant with strong mathematical abilities.",
        thinking_budget=10000
    )
    
    print(json.dumps(response, indent=2))
    return response


if __name__ == "__main__":
    # Run test when module is executed directly
    asyncio.run(test_claude_extended())
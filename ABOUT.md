# About OpenManus

OpenManus is an open-source AI agent framework developed by team members from MetaGPT. It provides a general-purpose AI agent capable of executing various tasks autonomously through a modular and extensible architecture.

## Core Purpose

OpenManus serves as an open alternative to the invite-only Manus AI platform, allowing anyone to create powerful AI agents without restrictions. The project aims to democratize access to autonomous AI technology by providing a comprehensive framework that can:

- Execute commands and run programs
- Browse the web and extract information
- Run Python code
- Perform complex reasoning tasks
- Interact with various APIs and services

## Architecture

OpenManus follows a modular architecture with several key components:

### 1. Agent Layer
- **BaseAgent**: Provides basic state management and execution loop
- **ReActAgent**: Implements the think-act pattern for sequential reasoning
- **ToolCallAgent**: Adds tool calling capabilities
- **Manus**: The primary agent that integrates multiple tools and capabilities

### 2. LLM Layer
- Provides an abstraction for interacting with large language models
- Supports various LLM providers (OpenAI, Anthropic, etc.)
- Handles context management, retries, and error handling
- Supports streaming responses

### 3. Memory Layer
- Manages conversation history and context
- Stores and retrieves messages for maintaining context

### 4. Tool Layer
- **BaseTool**: Abstract base class for all tools
- **BrowserUseTool**: For web interaction and automation
- **PythonExecute**: For executing Python code with safety measures
- **WebSearch**: For performing internet searches
- **StrReplaceEditor**: For editing text and files
- Various other specialized tools for different tasks

### 5. Prompt Layer
- System prompts define the agent's role and capabilities
- Task-specific prompts guide decision-making
- Dynamic prompt generation based on context

## Key Features

### Browser Automation
The BrowserUseTool provides comprehensive web interaction capabilities:
- Navigation (go to URL, back, refresh)
- Element interaction (click, input text, scroll)
- Content extraction (scrape information, extract structured data)
- Tab management (switch, open, close tabs)

### Python Execution
The PythonExecute tool allows executing Python code with:
- Timeout protection
- Isolated execution environment
- Output capture

### Claude 3.7 Extended Thinking Support
The claude_extended.py module provides integration with Anthropic's Claude 3.7 model's extended thinking capabilities:
- Configurable "thinking budget" to control token usage
- Support for complex reasoning tasks
- Fallback mechanisms to handle errors
- Compatible with both direct Anthropic API and OpenRouter

### Multi-Agent Collaboration
Through the Flow module, OpenManus supports multiple agents working together:
- Task assignment and coordination
- Result integration
- Flow control between agents

## Installation and Setup

OpenManus can be installed using either conda or uv (recommended):

### Using uv (Recommended):
```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone repository
git clone https://github.com/mannaandpoem/OpenManus.git
cd OpenManus

# Create virtual environment
uv venv --python 3.12
source .venv/bin/activate  # On Unix/macOS

# Install dependencies
uv pip install -r requirements.txt
```

### Configuration
1. Create a config.toml file in the config directory:
```bash
cp config/config.example.toml config/config.toml
```

2. Edit config/config.toml to add API keys and customize settings:
```toml
# Global LLM configuration
[llm]
model = "gpt-4o"
base_url = "https://api.openai.com/v1"
api_key = "sk-..."  # Your API key
max_tokens = 4096
temperature = 0.0
```

## Usage

### Basic Usage
```bash
python main.py
```

### MCP Tool Version
```bash
python run_mcp.py
```

### Multi-Agent Version
```bash
python run_flow.py
```

## Claude 3.7 Extended Thinking

The claude_extended.py module provides specialized support for Claude 3.7's extended thinking capabilities, which allows for:

1. More detailed step-by-step reasoning
2. Configurable token budget for complex tasks
3. Improved handling of complex reasoning tasks
4. Fallback mechanisms to ensure reliability

This integration allows OpenManus to leverage advanced reasoning capabilities for tasks requiring complex thinking while maintaining control over token usage.

## Future Developments

There are ongoing efforts to develop:

1. **OpenManus-RL**: A reinforcement learning-based approach for tuning LLM agents, developed collaboratively by researchers from UIUC and OpenManus
2. **OpenManus Online**: A web-based SaaS version of OpenManus being researched to make the technology more accessible without technical setup

## Community

OpenManus is built by contributors from MetaGPT and welcomes community participation. The project is available under an open-source license, making it accessible to developers and researchers worldwide.

## References

- [GitHub Repository](https://github.com/mannaandpoem/OpenManus)
- Inspired by projects like [anthropic-computer-use](https://github.com/anthropics/anthropic-quickstarts/tree/main/computer-use-demo) and [browser-use](https://github.com/browser-use/browser-use) 
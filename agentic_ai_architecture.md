# Agentic AI System Architecture: Calculator Agent

This diagram shows the specific implementation of the AI agent system using FastAPI, LangChain, and the Mistral 7B model for performing calculations.

```text
    User / Client (POST /chat)
          ↓
    FastAPI Endpoint (main.py)
          ↓
    LangChain Agent (ZERO_SHOT_REACT_DESCRIPTION)
          ↓
    Hugging Face Model (mistralai/Mistral-7B-Instruct-v0.2)
          ↓
    Tool Execution (llm-math / Calculator)
          ↓
    Final Response
```

## Component Details

**User / Client** - Sends a JSON request (e.g., `{"query": "What is 15 * 25?"}`) to the `/chat` endpoint.

**FastAPI Endpoint** - A web server implemented in `main.py` that handles the request and invokes the LangChain agent executor.

**LangChain Agent** - A "Reason-Act" (ReAct) agent that determines whether it needs to use a tool to answer the user's query based on the model's output.

**Hugging Face Model** - `Mistral-7B-Instruct-v0.2` acts as the reasoning engine to understand the query and decide on the tool usage.

**Tool Execution** - The `llm-math` tool (Calculator) which performs the actual mathematical operation requested by the user.

**Final Response** - The final calculated result (e.g., `{"input": "What is 15 * 25?", "output": "375"}`) returned to the client.

## Implementation Files
- **main.py**: The FastAPI application and agent logic.
- **test_agent.py**: Standalone test script for the agent.
- **requirements.txt**: Project dependencies.
- **.env**: Environment configuration (Hugging Face API Token).

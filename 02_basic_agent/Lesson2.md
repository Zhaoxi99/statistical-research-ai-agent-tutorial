# Lesson 02: Building Your First AI Agent

## Learning Objectives

In this lesson, we introduce the OpenAI Agents SDK and build our first AI agent in Python.

By the end of the lesson, you should be able to:

- Understand the difference between an LLM API call and an agent.
- Create an agent using the OpenAI Agents SDK.
- Execute agents using `Runner`.
- Build an interactive command-line chatbot.
- Understand conversational memory and context management.
- Compare manual conversation history with automatic session management.

## 1. From LLM APIs to AI Agents

In Lesson 01, we interacted with an LLM through the OpenAI Responses API.

For example:

```python
response = client.responses.create(
    model="gpt-5-mini",
    input="Explain Bayesian inference."
)
```

This makes a direct request to the model.

An AI agent extends this idea by combining an LLM with instructions and potentially external tools, memory, guardrails, and other capabilities.

The OpenAI Agents SDK provides a framework for implementing these systems.

At its core, an agent can be defined as:

```python
from agents import Agent

agent = Agent(
    name="Statistical Research Assistant",
    instructions="Help researchers understand statistical methods."
)
```

The `instructions` argument determines the agent's role and expected behaviour.

At this stage, our agent does not have external tools, so its functionality is similar to a conventional chatbot. We will introduce tool calling in Lesson 03.

## 2. Installation

Install the OpenAI Agents SDK:

```bash
python -m pip install openai-agents python-dotenv
```

As in Lesson 01, configure your `OPENAI_API_KEY` environment variable or provide it through a private `.env` file.

## 3. Creating and Running an Agent

The file `hello_agent.py` demonstrates how to create and execute a basic agent.

The main execution statement is:

```python
result = Runner.run_sync(agent, question)
```

`Runner` manages execution of the agent, including requests to the LLM and collection of its output.

The final response is available through:

```python
print(result.final_output)
```

Run the example:

```bash
python hello_agent.py
```

## 4. Building an Interactive Agent

Rather than defining the question directly in the script, we can use Python's `input()` function:

```python
question = input("You: ")
```

A `while` loop allows the user to ask multiple questions without restarting the program.

The file `interactive_agent.py` implements this functionality.

Run:

```bash
python interactive_agent.py
```

Type `exit` or `quit` to stop the program.

## 5. Conversational Memory

An LLM does not automatically remember previous interactions between independent API requests.

To implement conversational memory, we maintain a list of previous messages.

```python
history = []
```

After each agent run, we update this list:

```python
history = result.to_input_list()
```

Before the next run, the new user question is appended to the history.

This enables follow-up questions that depend on earlier conversation context.

Note that conversation history is not the same as modifying the LLM's learned parameters.

## 6. Automatic Memory Using Sessions

The OpenAI Agents SDK also supports session-based memory.

For example:

```python
from agents import SQLiteSession

session = SQLiteSession(
    "statistical_tutorial",
    "conversation_history.db"
)
```

We can then execute an agent with:

```python
result = Runner.run_sync(
    agent,
    question,
    session=session
)
```

The SDK automatically retrieves and stores conversation history associated with the session.

Unlike our manually managed list, the SQLite-backed session can retain conversation history across program restarts.

See `session_agent.py` for an implementation.

## 7. Experiments

**Experiment 1: Basic agent**

Run `hello_agent.py` and modify the question to investigate different statistical concepts.

**Experiment 2: Interactive agent**

Run `interactive_agent.py` and ask:

1. What is Bayesian linear regression?
2. How does it differ from frequentist regression?
3. Give me a mathematical example.

Observe whether the agent uses context from the preceding questions.

**Experiment 3: Conversational memory**

Restart `interactive_agent.py` and ask a follow-up question without providing the earlier context.

Compare this with the behaviour before restarting.

**Experiment 4: Persistent memory**

Run `session_agent.py`, ask several related questions, close the program, and restart it.

Investigate whether the conversation can continue from the previously stored context.

## 8. Limitations

The agents developed in this lesson do not yet have tools for accessing datasets or executing statistical computations.

Therefore, the agent may generate incorrect numerical answers despite being instructed to provide accurate information.

Conversational memory also increases the amount of context processed by the model, potentially increasing token usage and API costs.

Persistent sessions require appropriate management of locally stored data.

## 9. Key Takeaways

- The OpenAI Agents SDK simplifies the creation and execution of LLM-based agents.
- `Agent` defines an agent's configuration and instructions.
- `Runner` manages agent execution.
- Interactive applications can accept questions dynamically using `input()`.
- Conversation history provides context across multiple interactions.
- Sessions provide an alternative mechanism for managing conversation history.

In Lesson 03, we will extend the agent with Python statistical tools using function calling.

## References

- [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/)
- [Agents SDK Quickstart](https://openai.github.io/openai-agents-python/quickstart/)
- [Running Agents](https://openai.github.io/openai-agents-python/running_agents/)
- [Sessions](https://openai.github.io/openai-agents-python/sessions/)
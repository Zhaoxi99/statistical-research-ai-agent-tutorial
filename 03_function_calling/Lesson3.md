# Lesson 03: Function Calling — Connecting LLMs to Python Tools

## Learning Objectives

In Lesson 02, we built a conversational AI agent. In this lesson, we extend it with **Python functions for statistical computation**.

By the end of this lesson, you should be able to:

- Understand what function calling (tool calling) means and why it is useful.
- Write a reusable numerical function and expose it using `@function_tool`.
- Explain **how the LLM decides which tool to use**, including the role of function names, docstrings, type annotations, and agent instructions.
- Inspect the JSON Schema sent to the model.
- Verify that a tool was actually called, rather than trusting the final answer alone.
- Build an interactive tool-using agent and try a simple tool-description experiment.

No previous experience with JSON Schema is required.

## 1. Why Do AI Agents Need Tools?

LLMs can explain statistical ideas, but they are not guaranteed to perform numerical calculations accurately. For example, we might ask:

> Calculate the sample mean and standard deviation of 12, 15, 18, 20, and 25.

Instead of asking the LLM to calculate these numbers itself, we can provide it with a **Python function using NumPy**.

This separates two responsibilities:

- **LLM:** Understand the question, decide whether a calculation is needed, and explain the result.
- **Python:** Perform the actual numerical computation and return the values.

The model can still misinterpret or misreport a tool's result, so tool use improves reliability but does not guarantee correctness.

## 2. What Is Function Calling?

Function calling lets an LLM **request execution of a function that the developer has made available**.

A typical process is:

1. A user asks a question.
2. The LLM receives the question, the agent instructions, and descriptions of available tools.
3. The LLM decides whether a tool is useful and, if so, produces a structured call with arguments.
4. The OpenAI Agents SDK executes the corresponding Python function.
5. The tool returns its result to the model.
6. The model answers the question or requests another tool.

**Important:** Our LLM cannot freely run arbitrary Python code. It can request only the functions we deliberately register as tools.

## 3. Environment Setup

Install the dependencies in your virtual environment:

```bash
python -m pip install openai-agents numpy python-dotenv
```

Configure `OPENAI_API_KEY` as in the previous lessons, either through an environment variable or a private `.env` file. **Never commit a real API key to GitHub.** API calls may incur charges.

Files in this lesson:

```text
03_function_calling/
├── Lesson3.md
├── statistical_tools.py
├── statistical_agent.py
├── inspect_schema.py
├── interactive_agent.py
└── compare_descriptions.py
```

Run the examples from inside the `03_function_calling` directory.

## 4. Writing Reusable Statistical Functions

Before introducing an LLM, we should implement the underlying computation in ordinary Python.

The file `statistical_tools.py` defines `calculate_statistics(values)` and returns sample size, mean, sample standard deviation, minimum, and maximum.

For observations \(x_1,\ldots,x_n\), the sample mean is

\[
\bar{x}=\frac{1}{n}\sum_{i=1}^{n}x_i,
\]

and the sample standard deviation is

\[
s=\sqrt{\frac{1}{n-1}\sum_{i=1}^{n}(x_i-\bar{x})^2}.
\]

NumPy provides:

```python
np.mean(x)
np.std(x, ddof=1)
```

Here, `ddof=1` gives the denominator \(n-1\) in the sample variance.

Test the function without calling any LLM:

```bash
python statistical_tools.py
```

For 12, 15, 18, 20, and 25, the mean is **18** and the sample standard deviation is approximately **4.95**.

Notice that this function does not import `agents` or `openai`. This is good design: **the statistical calculation should be reusable and independently testable**.

## 5. Exposing a Python Function as an Agent Tool

In `statistical_agent.py`, we use the SDK's `@function_tool` decorator:

```python
from agents import function_tool
from statistical_tools import calculate_statistics

@function_tool
def describe_numbers(values: list[float]) -> dict:
    """Calculate sample size, mean, sample standard deviation, min and max.

    Use this tool for numerical descriptive statistics, not purely
    conceptual questions.

    Args:
        values: At least two finite numerical observations.
    """
    return calculate_statistics(values)
```

The runnable file also prints a debugging message and catches invalid input errors.

Two functions have different responsibilities:

| Function | Purpose |
| --- | --- |
| `calculate_statistics(values)` | Computes statistics using NumPy |
| `describe_numbers(values)` | Provides an interface that the LLM can call |

We register the tool when defining our agent:

```python
from agents import Agent, ModelSettings

agent = Agent(
    name="Statistical Research Assistant",
    instructions="Use statistical tools for numerical calculations.",
    tools=[describe_numbers],
    model_settings=ModelSettings(tool_choice="auto"),
)
```

`tool_choice="auto"` lets the model decide whether to request the tool. Registering it **does not guarantee that it will be used**. The runnable code creates the agent using `make_agent("clear")` so we can compare tool descriptions later.

## 6. How Does the LLM Know Which Tool to Use?

This is an important part of agent engineering. **The model generally does not receive your whole function's Python source code.** It receives *tool metadata* built by the SDK.

### 6.1 Function name

A name such as `describe_numbers` communicates more than a vague name such as `process_values`.

### 6.2 Ordinary comments versus docstrings

An ordinary comment helps programmers:

```python
def calculate_mean(values):
    # This function computes a mean.
    return sum(values) / len(values)
```

But the `#` comment inside the function is **not automatically provided to the LLM as a tool description**.

A **docstring** is different:

```python
def calculate_mean(values: list[float]) -> float:
    """Calculate the arithmetic mean of numerical observations.

    Use this tool when the user asks for an average or mean.

    Args:
        values: Numbers to average.
    """
    return sum(values) / len(values)
```

By default, the Agents SDK uses this docstring to construct tool and parameter descriptions. Thus, a tool docstring is not merely an explanation for other programmers: **it helps tell the LLM when and how to call the tool**.

### 6.3 Parameter names and type annotations

Consider:

```python
def describe_numbers(values: list[float]) -> dict:
```

- `values` tells the model what the input represents.
- `list[float]` tells the SDK that the input is an array of numbers.
- `-> dict` describes the Python return type; the *input* tool schema primarily concerns the arguments.

### 6.4 Agent instructions and the user question

The LLM also considers the agent's general instructions and the user's question. For example:

| Question | Expected behaviour |
| --- | --- |
| "Calculate the mean of 10, 20, and 30." | Request the statistical tool |
| "What does a mean represent?" | Answer without calling a numerical tool |
| "What are the minimum and maximum values?" | Request the statistical tool, if values are provided |
| "What is a p-value?" | No numerical tool required |

These are *expected* choices; the model may still make mistakes.

## 7. Understanding JSON Schema

When you decorate a function with `@function_tool`, the SDK generates a **JSON Schema** that defines its input arguments.

For `values: list[float]`, a simplified schema is:

```json
{
  "type": "object",
  "properties": {
    "values": {
      "type": "array",
      "items": {"type": "number"}
    }
  },
  "required": ["values"],
  "additionalProperties": false
}
```

The LLM can request a tool invocation with arguments such as:

```json
{"values": [12, 15, 18, 20, 25]}
```

The SDK then dispatches this call to the registered Python function.

**Try it yourself:** run:

```bash
python inspect_schema.py
```

It prints each tool's `.name`, `.description`, and `.params_json_schema`. Check which parts came from the function signature and docstring. You should not see its ordinary Python comments or full NumPy implementation automatically included.

## 8. Running the Interactive Statistical Agent

Run:

```bash
python interactive_agent.py
```

Ask:

> Calculate the mean and sample standard deviation of 12, 15, 18, 20, and 25.

You should see a debugging line similar to:

```text
TOOL EXECUTED: describe_numbers([12.0, 15.0, 18.0, 20.0, 25.0])
```

The script also inspects tool-request events from the SDK. This matters because **a correct-looking answer is not proof that the function was actually called**.

Next, ask:

> What is the difference between sample and population standard deviation?

The agent can explain this without calling the tool.

Our interactive agent retains the conversation context, as introduced in Lesson 02.

## 9. Reason–Act–Observe

Tool calling is a foundation of the **Reason–Act–Observe** pattern used by AI agents:

- **Reason:** Determine what action or information is needed.
- **Act:** Request a tool, such as `describe_numbers(values=[...])`.
- **Observe:** Receive the statistics computed by Python and use them to answer or decide what to do next.

The cycle can repeat before producing a final response. The SDK handles the model–tool interaction for us.

We can observe requests, arguments and results, but we should not assume that the model's private internal reasoning is exposed.

## 10. Experiment: Does Tool Documentation Affect Selection?

The runnable `statistical_agent.py` defines **two agent-facing tools** that call the **same underlying NumPy function**:

| Clear description | Vague description |
| --- | --- |
| `describe_numbers` | `process_values` |
| Clearly explains mean, sample SD, min/max and when to use the tool | Only says "Process some values" |

Run:

```bash
python compare_descriptions.py
```

The script creates two agents, each with one of these tools and otherwise comparable configurations. It asks numerical and conceptual questions and records whether the agent appropriately chose to call a tool. It saves the results to `tool_selection_results.csv`.

One useful metric is

\[
\text{Decision-match rate}=
\frac{\text{Runs with correct tool/no-tool decision}}{\text{Total runs}}.
\]

**Interpret cautiously:** Both the tool's **name** and **docstring** change in this experiment, so differences cannot be attributed solely to docstring quality. To isolate the latter, hold the model-facing tool name and every other condition fixed.

This is a small educational experiment rather than a performance benchmark. It evaluates whether a tool was *requested*, not whether the arguments were correct, the function succeeded or the final interpretation was accurate. The script makes real API calls.

## 11. Practice Exercises

**Exercise 1 — Numerical calculation:** Ask for the mean and sample SD of 12, 15, 18, 20, and 25. Check both the tool call and numerical result.

**Exercise 2 — Conceptual question:** Ask why sample variance uses \(n-1\). Does the agent avoid unnecessary computation?

**Exercise 3 — Conversational context:** Ask for the mean of 10, 20, and 30, then ask what happens when you add 100. Does the agent use the earlier values?

**Exercise 4 — Invalid input:** Ask for the sample standard deviation of a single observation. Does the tool return an error rather than a fabricated number?

**Exercise 5 — Documentation:** Change the docstring of `describe_numbers`, rerun `inspect_schema.py`, and observe what changes in the metadata. Then try the comparison experiment again.

## 12. Limitations and Common Problems

- **Tool selection is imperfect:** The LLM may call the wrong tool or avoid a needed tool.
- **Tool arguments may be wrong:** Input validation remains necessary.
- **Tools can fail:** For example, a statistical function might receive too few observations.
- **The final answer can still be wrong:** The LLM may misinterpret or misreport valid tool output.
- **Tool schemas matter:** A function such as `def inspect_data(df)` without a type annotation may cause schema errors. A pandas DataFrame is also not an appropriate model-generated JSON argument; we'll address dataset selection in Lesson 04.
- **Current capabilities are limited:** The agent does not yet load CSV datasets or fit regression models.

For a reliable research application, **tool selection, tool execution, and final statistical interpretation need separate checks**.

## 13. Key Takeaways

- Function calling connects an LLM to selected Python capabilities.
- `@function_tool` exposes functions through the OpenAI Agents SDK.
- Function names, **docstrings**, parameter types, agent instructions, and user questions all influence tool selection.
- Ordinary `#` comments inside a function are not automatically visible to the LLM.
- Numerical calculations should be separate from the LLM-facing tool interface.
- Observe actual tool calls rather than trusting plausible final answers.

In **Lesson 04: Data Analysis**, we will move from manually entered numbers to CSV datasets, pandas DataFrames, regression models and statistical diagnostics.

## References

- [OpenAI Agents SDK — Tools](https://openai.github.io/openai-agents-python/tools/)
- [OpenAI Agents SDK — Running Agents](https://openai.github.io/openai-agents-python/running_agents/)
- [OpenAI API — Function Calling](https://platform.openai.com/docs/guides/function-calling)
- [NumPy Documentation](https://numpy.org/doc/stable/)

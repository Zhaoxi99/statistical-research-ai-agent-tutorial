"""Privacy-conscious local monitoring helpers (no question/answer text logged)."""
from __future__ import annotations
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from agents import Runner, RunConfig, ToolCallItem, ToolCallOutputItem, trace

HERE = Path(__file__).resolve().parent
LOG_PATH = HERE / 'agent_runs.jsonl'


def run_monitored(agent, question: str, *, config_id: str = 'baseline', log_path: Path = LOG_PATH):
    """Execute one agent run, append a minimal record, and return the full result.

    Logs intentionally omit user prompts, generated answers and tool arguments.
    """
    start = time.perf_counter()
    result = None
    failure = None
    try:
        with trace(f'research-agent-{config_id}'):
            result = Runner.run_sync(
                agent, question, max_turns=12,
                run_config=RunConfig(trace_include_sensitive_data=False),
            )
        return result
    except Exception as exc:
        failure = type(exc).__name__
        raise
    finally:
        calls, outputs = [], []
        if result is not None:
            for item in result.new_items:
                if isinstance(item, ToolCallItem):
                    calls.append(getattr(item.raw_item, 'name', 'unknown'))
                elif isinstance(item, ToolCallOutputItem):
                    outputs.append(item.output)
        usage = getattr(getattr(result, 'context_wrapper', None), 'usage', None)
        record = {
            'timestamp_utc': datetime.now(timezone.utc).isoformat(),
            'config_id': config_id,
            'agent_name': agent.name,
            'model': agent.model if isinstance(agent.model, str) else 'SDK default/configured model',
            'status': 'failed' if failure else 'success',
            'error_type': failure,
            'latency_seconds': round(time.perf_counter() - start, 3),
            'tool_names': calls,
            'tool_calls': len(calls),
            'tool_outputs': len(outputs),
            'input_tokens': getattr(usage, 'input_tokens', None),
            'output_tokens': getattr(usage, 'output_tokens', None),
            'total_tokens': getattr(usage, 'total_tokens', None),
        }
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open('a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

"""No API calls: smoke test logging format and summary calculation."""
import json
import tempfile
from pathlib import Path
from summarize_runs import summarize


def main():
    with tempfile.TemporaryDirectory() as temp:
        path = Path(temp) / 'runs.jsonl'
        example = {
            'timestamp_utc': '2026-10-18T12:00:00+00:00',
            'config_id': 'example', 'agent_name': 'Test Agent',
            'model': 'test', 'status': 'success', 'error_type': None,
            'latency_seconds': 1.25, 'tool_names': ['inspect_data'],
            'tool_calls': 1, 'tool_outputs': 1,
            'input_tokens': 100, 'output_tokens': 50, 'total_tokens': 150,
        }
        path.write_text(json.dumps(example) + '\n', encoding='utf-8')
        result = summarize(path)
        assert result['example']['runs'] == 1
        assert result['example']['success_rate'] == 1.0
        assert result['example']['mean_total_tokens'] == 150
        assert result['example']['tool_counts'] == {'inspect_data': 1}
        assert 'question' not in example and 'answer' not in example
    print('\nPASS: offline log aggregation and privacy fields')


if __name__ == '__main__':
    main()

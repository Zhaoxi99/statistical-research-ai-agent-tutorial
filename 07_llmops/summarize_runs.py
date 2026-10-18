"""Summarize saved local telemetry without making API requests."""
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median

HERE = Path(__file__).resolve().parent


def summarize(path: Path):
    if not path.exists():
        print(f'No log at {path}. First run: python run_monitored.py')
        return {}
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    if not rows:
        print('Log is empty.')
        return {}
    groups = defaultdict(list)
    for row in rows:
        groups[row['config_id']].append(row)
    summary = {}
    for name, entries in sorted(groups.items()):
        latencies = [r['latency_seconds'] for r in entries]
        known_tokens = [r['total_tokens'] for r in entries if isinstance(r.get('total_tokens'), (int, float))]
        tool_counts = Counter(tool for r in entries for tool in r['tool_names'])
        values = {
            'runs': len(entries),
            'success_rate': sum(r['status'] == 'success' for r in entries) / len(entries),
            'mean_latency_seconds': round(mean(latencies), 3),
            'median_latency_seconds': round(median(latencies), 3),
            'mean_total_tokens': round(mean(known_tokens), 1) if known_tokens else None,
            'token_records_available': len(known_tokens),
            'tool_counts': dict(tool_counts),
        }
        summary[name] = values
        print(f'\n{name}:')
        for key, value in values.items():
            print(f'  {key}: {value}')
    return summary


if __name__ == '__main__':
    summarize(HERE / 'agent_runs.jsonl')

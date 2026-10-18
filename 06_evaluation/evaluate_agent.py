"""Live evaluation: costs API tokens. Run from this directory.

Scores tool choice separately from execution success and human-graded answer quality.
"""
import argparse
import csv
import json
from pathlib import Path
from datetime import datetime, timezone
from agents import Runner, ToolCallItem, ToolCallOutputItem
from research_agent import agent
from evaluation_cases import CASES

HERE = Path(__file__).resolve().parent


def extract_events(result):
    """Read chronological tool events; don't interpret output text as a tool call."""
    calls, outputs = [], []
    for item in result.new_items:
        if isinstance(item, ToolCallItem):
            calls.append(getattr(item.raw_item, 'name', 'unknown'))
        elif isinstance(item, ToolCallOutputItem):
            raw = item.output
            try:
                parsed = json.loads(raw) if isinstance(raw, str) else raw
            except (ValueError, TypeError):
                parsed = raw
            outputs.append(parsed)
    return calls, outputs


def evaluate_case(case):
    record = {"id": case['id'], "question": case['question'], "required": case['required'],
              "allowed": case['allowed'], "expect_error": case.get('expect_error', False),
              "calls": [], "tool_outputs": [], "answer": "", "run_error": "",
              "required_recall": 0.0, "unexpected_calls": [], "tool_error_count": 0,
              "decision_pass": False, "tool_execution_pass": False}
    try:
        result = Runner.run_sync(agent, case['question'], max_turns=12)
        calls, outputs = extract_events(result)
        record['calls'] = calls
        record['tool_outputs'] = outputs
        record['answer'] = str(result.final_output)
        required, allowed, actual = set(case['required']), set(case['allowed']), set(calls)
        record['required_recall'] = len(required & actual) / len(required) if required else 1.0
        record['unexpected_calls'] = sorted(actual - allowed)
        record['decision_pass'] = required <= actual and actual <= allowed
        record['tool_error_count'] = sum(
            isinstance(o, dict) and bool(o.get('error')) for o in outputs
        )
        # A deliberate invalid-input test should produce an error, without fabricating success.
        record['tool_execution_pass'] = (
            record['tool_error_count'] > 0 if case.get('expect_error')
            else record['tool_error_count'] == 0 and len(calls) == len(outputs)
        )
    except Exception as exc:
        record['run_error'] = f'{type(exc).__name__}: {exc}'
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repeats', type=int, default=1, help='Independent repetitions per question')
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be >= 1')
    records = []
    for rep in range(1, args.repeats + 1):
        for case in CASES:
            print(f"Running {case['id']} (repeat {rep})...", flush=True)
            record = evaluate_case(case)
            record['repeat'] = rep
            records.append(record)
            print('  calls:', record['calls'], 'decision pass:', record['decision_pass'],
                  'execution pass:', record['tool_execution_pass'])
    output = HERE / 'evaluation_results.jsonl'
    with output.open('w', encoding='utf-8') as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + '\n')
    fields = ['id', 'repeat', 'required_recall', 'decision_pass', 'tool_execution_pass',
              'tool_error_count', 'run_error', 'calls', 'unexpected_calls', 'answer']
    with (HERE / 'evaluation_summary.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in records:
            w.writerow({k: json.dumps(r[k], ensure_ascii=False) if isinstance(r[k], list) else r[k] for k in fields})
    n = len(records)
    print('\nSUMMARY')
    print(f"Decision pass: {sum(r['decision_pass'] for r in records)}/{n}")
    print(f"Execution pass: {sum(r['tool_execution_pass'] for r in records)}/{n}")
    print('Saved evaluation_results.jsonl and evaluation_summary.csv')
    print('Note: read final answers and score interpretation manually; these scores DO NOT assess factual correctness.')

if __name__ == '__main__':
    main()

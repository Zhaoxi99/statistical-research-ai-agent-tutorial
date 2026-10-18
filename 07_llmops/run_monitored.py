"""Monitor multiple independent executions of the same research question."""
import argparse
from dotenv import load_dotenv
from research_agent import agent
from monitoring import run_monitored

load_dotenv()

QUESTION = (
    'Investigate the association of age with biomarker, adjusting '
    'for sex, and check regression assumptions in biomarker.'
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repeats', type=int, default=1,
                        help='Number of independent monitored runs (default: 1)')
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be at least 1')

    for i in range(1, args.repeats + 1):
        print(f'\nRunning baseline: {i}/{args.repeats}')
        try:
            result = run_monitored(agent, QUESTION, config_id='baseline')
            print('AGENT ANSWER:\n', result.final_output)
        except Exception as exc:
            print(f'Run {i} failed: {type(exc).__name__}: {exc}')

    print('\nRun metadata appended to agent_runs.jsonl.')
    print('Use: python summarize_runs.py')


if __name__ == '__main__':
    main()

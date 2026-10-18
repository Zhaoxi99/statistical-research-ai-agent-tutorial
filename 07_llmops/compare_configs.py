"""Compare two instruction configurations with repeated independent runs.

This compares operational metrics, not statistical answer correctness.
"""
import argparse
from dataclasses import replace
from dotenv import load_dotenv
from research_agent import agent
from monitoring import run_monitored

load_dotenv()

QUESTION = 'Fit biomarker on age and sex in the biomarker dataset, then check assumptions.'
CONFIGS = {
    'baseline': agent,
    'concise': replace(agent, instructions=(
        'You are a statistical research assistant. Use the provided tools for numerical '
        'results. Do not invent data. Report association, not causation. '
        'Answer concisely while stating limitations.'
    )),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repeats', type=int, default=1,
                        help='Independent runs per configuration (default: 1)')
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be at least 1')

    # Alternate configurations by repetition to reduce order/time confounding.
    for i in range(1, args.repeats + 1):
        for config_id, selected_agent in CONFIGS.items():
            print(f'\nRunning {config_id}: {i}/{args.repeats}')
            try:
                result = run_monitored(selected_agent, QUESTION, config_id=config_id)
                print(result.final_output)
            except Exception as exc:
                print(f'Run failed: {type(exc).__name__}: {exc}')

    print('\nNow run: python summarize_runs.py')


if __name__ == '__main__':
    main()

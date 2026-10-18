"""Stage 2: show bounded task-dependent specialist configurations."""
import argparse
from specialists.factory import choose_roles,make_specialist
from dotenv import load_dotenv
load_dotenv()
p=argparse.ArgumentParser(); p.add_argument('--live',action='store_true'); args=p.parse_args()
for task in ['Fit biomarker regression','Find literature on statistical methods','Review biomarker research evidence']:
    roles=choose_roles(task)
    print('\nTASK:',task,'\nROLES:',roles)
    if args.live:
        agents=[make_specialist(role) for role in roles]
        for agent in agents:print('  configured:',agent.name,'tools:',len(agent.tools))
if args.live:
    from agents import Runner
    specialist=make_specialist('statistics')
    result=Runner.run_sync(specialist,'Inspect the biomarker dataset and report the mean age.')
    print('\nLIVE RESULT:\n',result.final_output)

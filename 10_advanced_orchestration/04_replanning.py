"""Stage 4: a deterministic replanning rule; no unbounded autonomous loops."""
from planning import ResearchPlan,Step,example_plan
from orchestration import Outcome

def revise_plan(original:ResearchPlan,results:dict[str,Outcome]) -> ResearchPlan:
    """One bounded remediation attempt when literature step failed."""
    if results.get('literature',Outcome('missing','')).status!='failed':
        return original
    # Show a clear human-visible alternative, without automatically looping.
    return ResearchPlan(goal='Recover from failed literature retrieval',steps=[
        Step(id='retry_literature',objective='Retry literature retrieval with a narrower search query',role='literature'),
        Step(id='review',objective='Review retrieved evidence or report remaining limitations',role='reviewer',depends_on=['retry_literature'])])

if __name__=='__main__':
    plan=example_plan()
    failed={'literature':Outcome('failed','Search service unavailable')}
    print('Original:',[s.id for s in plan.steps])
    print('Revised:',revise_plan(plan,failed).model_dump_json(indent=2))
    print('This demo produces ONE revised plan. It does not repeat indefinitely.')

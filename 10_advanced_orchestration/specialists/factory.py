"""Create specialists from a controlled role catalogue, never from arbitrary user text."""

ROLE_CATALOG = {
    'statistics': 'Analyse the bundled datasets with tools; never invent numerical results or causal claims.',
    'literature': 'Find methodological research. Cite real URLs, distinguish evidence from speculation.',
    'reviewer': 'Critically review supplied findings for missing evidence and unsupported claims. Do not invent computations.'
}

def choose_roles(request: str) -> list[str]:
    """Offline deterministic example of task-dependent role configuration."""
    q=request.lower()
    roles=[]
    if any(w in q for w in ('regression','biomarker','dataset','statistical','diagnostic')):
        roles.append('statistics')
    if any(w in q for w in ('literature','research','paper','evidence','methodolog')):
        roles.append('literature')
    if any(w in q for w in ('review','critique','validate')):
        roles.append('reviewer')
    return roles or ['literature']

def make_specialist(role: str, use_qdrant: bool = False):
    from agents import Agent, WebSearchTool
    from tools.agent_tools import inspect_data, describe_variable, fit_regression, check_regression_assumptions
    from tools.retrieval_tools import search_indexed_literature
    if role not in ROLE_CATALOG:
        raise ValueError(f'Unknown specialist role: {role}')
    tools=[]
    if role == 'statistics':
        tools=[inspect_data, describe_variable, fit_regression, check_regression_assumptions]
    elif role == 'literature':
        tools=[WebSearchTool(search_context_size='medium')]
        if use_qdrant:
            tools.append(search_indexed_literature)
    return Agent(name=role.title()+' Specialist',instructions=ROLE_CATALOG[role],tools=tools)

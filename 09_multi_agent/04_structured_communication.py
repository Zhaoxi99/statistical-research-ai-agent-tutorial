"""Structured specialist outputs, validated by Pydantic, then merged by a manager."""
from dotenv import load_dotenv
from agents import Agent,Runner
from specialists.schemas import StatisticalFinding,LiteratureFinding
from tools.agent_tools import inspect_data,fit_regression
from specialists.literature_agent import make_literature_agent

def main():
    load_dotenv()
    stats=Agent(name='Structured Statistical Specialist',output_type=StatisticalFinding,
        instructions=('Use tools to fit biomarker ~ age + sex in dataset biomarker. '
                      'Fill all schema fields strictly from tool results; limitations must note observational association is not causation.'),
        tools=[inspect_data,fit_regression])
    findings=Runner.run_sync(stats,'Inspect and fit biomarker on age and sex in biomarker.').final_output
    # Show validation and explicit serialization. This object can be passed to downstream code.
    print('Structured statistical finding:\n',findings.model_dump_json(indent=2))
    literature=make_literature_agent()
    lit_result=Runner.run_sync(literature,'Find methodological sources on assumptions and limitations of OLS regression. Provide URLs.').final_output
    # A separate LLM extracts a structured result from the source-grounded specialist answer.
    extractor=Agent(name='Evidence Formatter',output_type=LiteratureFinding,
        instructions=('Structure the supplied literature answer. Include only cited URLs actually present in the input. Do not invent citations.'))
    lit=Runner.run_sync(extractor,str(lit_result)).final_output
    print('\nStructured literature finding:\n',lit.model_dump_json(indent=2))
    manager=Agent(name='Report Writer',instructions='Write a concise report based ONLY on the supplied specialist findings. Do not invent facts or URLs.')
    combined=f'STATISTICS:\n{findings.model_dump_json()}\n\nLITERATURE:\n{lit.model_dump_json()}'
    report=Runner.run_sync(manager,combined).final_output
    print('\nFINAL REPORT\n',report)

if __name__=='__main__':main()

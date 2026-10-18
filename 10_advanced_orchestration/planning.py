"""Validated, bounded plans. Plan objects can be tested without an LLM."""
from pydantic import BaseModel, Field, model_validator
from specialists.factory import ROLE_CATALOG

class Step(BaseModel):
    id: str = Field(min_length=1)
    objective: str = Field(min_length=5)
    role: str
    depends_on: list[str] = Field(default_factory=list)

class ResearchPlan(BaseModel):
    goal: str = Field(min_length=5)
    steps: list[Step] = Field(min_length=1, max_length=6)
    @model_validator(mode='after')
    def check_dag(self):
        seen=set()
        for step in self.steps:
            if step.id in seen: raise ValueError('Duplicate step id')
            if step.role not in ROLE_CATALOG: raise ValueError('Unapproved role')
            if not set(step.depends_on).issubset(seen):
                raise ValueError('Dependencies must refer to earlier steps')
            seen.add(step.id)
        return self

def example_plan() -> ResearchPlan:
    return ResearchPlan(goal='Analyse the biomarker dataset and contextualise the findings',steps=[
        Step(id='analysis',objective='Inspect data, fit OLS and check assumptions',role='statistics'),
        Step(id='literature',objective='Retrieve relevant regression diagnostic literature',role='literature'),
        Step(id='review',objective='Review evidence and note any limitations',role='reviewer',depends_on=['analysis','literature'])])

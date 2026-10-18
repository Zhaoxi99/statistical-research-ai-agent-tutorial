"""Offline regression tests: no OpenAI calls or remote database required."""
import asyncio,unittest
from pydantic import ValidationError
from planning import Step,ResearchPlan,example_plan
from orchestration import execute_plan
from specialists.factory import choose_roles
from importlib import import_module

class TestOrchestration(unittest.TestCase):
    def test_plan(self):
        p=example_plan();self.assertEqual(len(p.steps),3)
    def test_forward_dependency_rejected(self):
        with self.assertRaises(ValidationError):
            ResearchPlan(goal='Example research goal',steps=[Step(id='a',objective='Do the first task',role='statistics',depends_on=['b'])])
    def test_role_is_restricted(self):
        with self.assertRaises(ValidationError):
            ResearchPlan(goal='Example research goal',steps=[Step(id='a',objective='Run a Python script',role='arbitrary_shell')])
    def test_role_selection(self):
        self.assertEqual(choose_roles('Fit biomarker regression'),['statistics'])
    def test_parallel_waves(self):
        async def worker(step,prior):return step.id
        result,waves=asyncio.run(execute_plan(example_plan(),worker))
        self.assertEqual(waves,[['analysis','literature'],['review']])
        self.assertTrue(all(x.status=='ok' for x in result.values()))
    def test_failure_skips_dependent_step(self):
        async def worker(step,prior):
            if step.id=='analysis':raise RuntimeError('example failure')
            return 'ok'
        result,_=asyncio.run(execute_plan(example_plan(),worker))
        self.assertEqual(result['analysis'].status,'failed')
        self.assertEqual(result['review'].status,'skipped')
    def test_replanning_bounded(self):
        module=import_module('04_replanning')
        from orchestration import Outcome
        p=module.revise_plan(example_plan(),{'literature':Outcome('failed','offline')})
        self.assertEqual(len(p.steps),2)
if __name__=='__main__':unittest.main(verbosity=2)

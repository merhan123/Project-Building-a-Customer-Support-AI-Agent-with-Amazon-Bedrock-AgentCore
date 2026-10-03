"""Offline behavior checks. These are not substitutes for deployed test evidence."""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch
from types import SimpleNamespace

os.environ['AWS_ACCESS_KEY_ID'] = 'offline-test'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'offline-test'
os.environ['AWS_EC2_METADATA_DISABLED'] = 'true'
spec = importlib.util.spec_from_file_location('support_agent', Path(__file__).resolve().parents[1] / 'starter/main.py')
agent = importlib.util.module_from_spec(spec)
spec.loader.exec_module(agent)


class AgentTests(unittest.TestCase):
    def calculate(self, points, tier, total, category='standard'):
        interpreter = MagicMock()
        def execute(method, args):
            self.assertEqual(method, 'executeCode')
            self.assertTrue(args['clearContext'])
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                exec(args['code'], {})
            return {'stream': iter([{'result': {'content': [{'type': 'text', 'text': out.getvalue()}]}}])}
        interpreter.invoke.side_effect = execute
        with patch.object(agent, 'code_session') as session:
            session.return_value.__enter__.return_value = interpreter
            raw = json.loads(agent.calculate_loyalty_discount(points, tier, total, category))
        return raw

    def test_course_discount(self):
        r = self.calculate(4250, 'Gold', 150)
        self.assertEqual(r['points_redeemed'], 4000)
        self.assertEqual(r['tier_discount_pct'], 10)
        self.assertEqual(r['tier_discount'], '11.00')
        self.assertEqual(r['final_total'], '99.00')
        self.assertEqual(r['remaining_points'], 250)
        self.assertEqual(r['points_earned'], 99)

    def test_caps_rounding_and_earning(self):
        self.assertEqual(self.calculate(10000, 'Silver', 19.99)['points_redeemed'], 500)
        self.assertEqual(self.calculate(499, 'Silver', 100)['points_redeemed'], 0)
        self.assertEqual(self.calculate(500, 'Platinum', 0)['final_total'], '0.00')
        r = self.calculate(0, 'Gold', 10.05, 'device')
        self.assertEqual(r['tier_discount'], '1.01')
        self.assertEqual(r['points_earned'], 18)
        self.assertEqual(self.calculate(0, 'Silver', 10, 'fresh')['points_earned'], 50)

    def test_invalid_inputs_and_labeled_fallback(self):
        with patch.object(agent, 'code_session', side_effect=RuntimeError('sandbox unavailable')):
            for args in [(-1, 'Gold', 10), (0, 'Invalid', 10), (0, 'Gold', -10), (0, 'Gold', float('nan'))]:
                self.assertIn('error', json.loads(agent.calculate_loyalty_discount(*args)))
            result = json.loads(agent.calculate_loyalty_discount(4250, 'Gold', 150))
            self.assertEqual(result['calculation_source'], 'tier_only_fallback')
            self.assertEqual(result['final_total'], '135.00')
            self.assertEqual(result['points_redeemed'], 0)
            self.assertEqual(result['tier_discount_pct'], 10)
            self.assertEqual(result['remaining_points'], 4250)

    def test_namespace_compatibility(self):
        client = MagicMock()
        client.get_memory_strategies.return_value = [
            {'type': 'SEMANTIC', 'namespaceTemplates': ['cs_agent/{actorId}/facts']},
            {'type': 'USER_PREFERENCE', 'namespaces': ['cs_agent/{actorId}/preferences']}]
        self.assertEqual(len(agent.get_namespaces(client, 'mem')), 2)

    def test_memory_does_not_save_injected_context_or_tool_results(self):
        client = MagicMock()
        client.get_memory_strategies.return_value = [{'type': 'SEMANTIC', 'namespaceTemplates': ['cs_agent/{actorId}/facts']}]
        client.retrieve_memories.return_value = [{'content': {'text': 'Name: Jane'}}]
        hook = agent.MemoryHook('CUST-123', 's-A', client, 'mem')
        message = {'role': 'user', 'content': [{'text': 'Hello'}]}
        hook.retrieve_customer_context(SimpleNamespace(message=message))
        self.assertIn('Customer Context', message['content'][0]['text'])
        client.retrieve_memories.assert_called_once_with(memory_id='mem', namespace='cs_agent/CUST-123/facts', query='Hello', top_k=5)
        result = {'role': 'user', 'content': [{'toolResult': {'content': [{'text': 'irrelevant'}]}}]}
        hook.retrieve_customer_context(SimpleNamespace(message=result))
        self.assertEqual(client.retrieve_memories.call_count, 1)
        hook.save_support_interaction(SimpleNamespace(agent=SimpleNamespace(messages=[message, result, {'role': 'assistant', 'content': [{'text': 'Hi!'}]}])))
        self.assertEqual(client.create_event.call_args.kwargs['messages'], [('Hello', 'USER'), ('Hi!', 'ASSISTANT')])

    def test_rag_empty_and_grounded_chunks(self):
        with patch.object(agent, 'KB_ID', 'ABCDEFGHIJ'), patch.object(agent, '_bedrock_runtime') as kb:
            kb.retrieve.return_value = {'retrievalResults': []}
            self.assertIn('No relevant', agent.search_knowledge_base('test'))
            kb.retrieve.return_value = {'retrievalResults': [{'content': {'text': 'First'}}, {'content': {'text': 'Second'}}]}
            self.assertEqual(agent.search_knowledge_base('test'), 'First\n---\nSecond')


if __name__ == '__main__':
    unittest.main()

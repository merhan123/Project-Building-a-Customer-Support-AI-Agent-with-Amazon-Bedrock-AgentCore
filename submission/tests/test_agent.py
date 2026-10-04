"""Offline behavior checks. These are not substitutes for deployed test evidence."""
import asyncio
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
spec = importlib.util.spec_from_file_location('support_agent', Path(__file__).resolve().parents[1] / 'main.py')
agent = importlib.util.module_from_spec(spec)
spec.loader.exec_module(agent)


class ToolPage(list):
    def __init__(self, tools, pagination_token=None):
        super().__init__(tools)
        self.pagination_token = pagination_token


class AgentTests(unittest.TestCase):
    def test_gateway_failures_are_safe_and_actionable(self):
        secret = 'https://dummy.invalid/mcp?token=DO_NOT_EXPOSE'
        cases = [('connect', ConnectionError(secret), 'connection failed'),
                 ('load', TimeoutError(secret), 'timed out'),
                 ('load', RuntimeError(secret), 'tool loading failed'),
                 ('empty', None, 'tool loading failed'),
                 ('second_page', ConnectionError(secret), 'connection failed')]
        for stage, error, expected in cases:
            with self.subTest(stage=stage, expected=expected), contextlib.ExitStack() as stack:
                stack.enter_context(patch.object(agent, 'GATEWAY_URL', secret))
                browser = stack.enter_context(patch.object(agent, 'AgentCoreBrowser')).return_value
                stack.enter_context(patch.object(agent, 'MemoryHook'))
                stack.enter_context(patch.object(agent, 'session_history', return_value=[]))
                factory = stack.enter_context(patch.object(agent, 'MCPClient'))
                model_agent = stack.enter_context(patch.object(agent, 'Agent'))
                gateway = factory.return_value.__enter__.return_value
                if stage == 'connect':
                    factory.return_value.__enter__.side_effect = error
                elif stage == 'empty':
                    gateway.list_tools_sync.return_value = ToolPage([])
                elif stage == 'second_page':
                    gateway.list_tools_sync.side_effect = [ToolPage(['tool'], 'next'), error]
                else:
                    gateway.list_tools_sync.side_effect = error
                with self.assertLogs(agent.logger, level='ERROR') as logs:
                    result = asyncio.run(agent.invoke({'prompt': 'Track my order'}))
                self.assertIn(expected, result['error'])
                self.assertIn('try again later', result['error'])
                self.assertNotIn(secret, json.dumps(result) + str(logs.output))
                self.assertNotIn('DO_NOT_EXPOSE', str(logs.output))
                model_agent.assert_not_called()
                browser.close_platform.assert_called_once()
                if stage != 'connect':
                    factory.return_value.__exit__.assert_called_once()

    def test_gateway_success_loads_all_pages_and_logs_count(self):
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(agent, 'AgentCoreBrowser'))
            hook = stack.enter_context(patch.object(agent, 'MemoryHook')).return_value
            hook.warnings = []
            stack.enter_context(patch.object(agent, 'session_history', return_value=[]))
            factory = stack.enter_context(patch.object(agent, 'MCPClient'))
            gateway = factory.return_value.__enter__.return_value
            gateway.list_tools_sync.side_effect = [ToolPage(['api'], 'next'), ToolPage(['lambda'])]
            model_agent = stack.enter_context(patch.object(agent, 'Agent'))
            from unittest.mock import AsyncMock
            model_agent.return_value.invoke_async = AsyncMock()
            model_agent.return_value.messages = [{'role': 'assistant', 'content': [{'text': 'Done'}]}]
            with self.assertLogs(agent.logger, level='INFO') as logs:
                result = asyncio.run(agent.invoke({'prompt': 'Track my order'}))
            self.assertEqual(result, 'Done')
            self.assertIn('Loaded 2 tools', str(logs.output))
            self.assertEqual(model_agent.call_args.kwargs['tools'][-2:], ['api', 'lambda'])
            gateway.list_tools_sync.assert_called_with(pagination_token='next')

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

    def test_incomplete_sandbox_result_uses_labeled_fallback(self):
        interpreter = MagicMock()
        interpreter.invoke.return_value = {'stream': iter([{'result': {'content': [
            {'type': 'text', 'text': '{"final_total": "0.00"}'}]}}])}
        with patch.object(agent, 'code_session') as session, self.assertLogs(agent.logger, level='ERROR'):
            session.return_value.__enter__.return_value = interpreter
            result = json.loads(agent.calculate_loyalty_discount(4250, 'Gold', 150))
        self.assertEqual(result['calculation_source'], 'tier_only_fallback')
        self.assertEqual(result['final_total'], '135.00')
        self.assertEqual(result['tier_discount_pct'], 10)
        self.assertEqual(result['remaining_points'], 4250)

    def test_display_and_memory_preserve_calculator_result(self):
        client = MagicMock()
        client.get_memory_strategies.return_value = []
        hook = agent.MemoryHook('CUST-123', 'test', client, 'mem')
        data = self.calculate(4250, 'Gold', 150)
        messages = [
            {'role': 'user', 'content': [{'text': 'Calculate my discount'}]},
            {'role': 'assistant', 'content': [{'toolUse': {'toolUseId': 'calc', 'name': 'calculate_loyalty_discount'}}]},
            {'role': 'user', 'content': [{'toolResult': {'toolUseId': 'calc', 'status': 'success', 'content': [{'text': json.dumps(data)}]}}]},
            {'role': 'assistant', 'content': [{'text': 'Incorrect total: $95.00'}]}]
        hook.save_support_interaction(SimpleNamespace(agent=SimpleNamespace(messages=messages)))
        displayed = agent.plain_text(messages[-1])
        self.assertIn('Final total: $99.00', displayed)
        self.assertNotIn('$95.00', displayed)
        self.assertEqual(client.create_event.call_args.kwargs['messages'][1], (displayed, 'ASSISTANT'))
        messages += [{'role': 'user', 'content': [{'text': 'Hello'}]}, {'role': 'assistant', 'content': [{'text': 'Hi'}]}]
        agent.ground_calculation_response(messages)
        self.assertEqual(agent.plain_text(messages[-1]), 'Hi')

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

"""Verify the real Gateway and catalog before deployment."""
import json
from pathlib import Path
import boto3
from mcp.client.streamable_http import streamable_http_client
from strands.tools.mcp.mcp_client import MCPClient

root = Path(__file__).resolve().parent
state = json.loads((root / 'resources.json').read_text())
session = boto3.Session(profile_name='udacity', region_name=state['region'])
report = {'account': session.client('sts').get_caller_identity()['Account']}
with MCPClient(lambda: streamable_http_client(state['gateway_url'])) as client:
    tools = client.list_tools_sync()
    report['gateway_tools'] = [tool.tool_name for tool in tools]
    assert len(tools) == 6, report['gateway_tools']
    order_tool = next(t.tool_name for t in tools if t.tool_name.endswith('___get_order'))
    report['order_lookup'] = client.call_tool_sync('setup-order', order_tool, {'order_id': 'ORD-001'})
    assert not report['order_lookup'].get('isError'), report['order_lookup']
response = session.client('bedrock-agent-runtime').retrieve(knowledgeBaseId=state['kb_id'], retrievalQuery={'text': 'What is the return policy for electronics?'})
report['knowledge_base_results'] = response.get('retrievalResults', [])
assert any('15' in r.get('content', {}).get('text', '') for r in report['knowledge_base_results']), report
report['memory_status'] = session.client('bedrock-agentcore-control').get_memory(memoryId=state['memory_id'])['memory']['status']
assert report['memory_status'] == 'ACTIVE'
(root / 'evidence/setup-verification.json').write_text(json.dumps(report, indent=2, default=str) + '\n')
print(json.dumps(report, indent=2, default=str))

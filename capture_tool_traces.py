"""Save complete deployed responses; agentcore CLI displays only the response field."""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import uuid
import boto3
from run_submission_tests import SCENARIOS

root = Path(__file__).resolve().parent
state = json.loads((root / 'resources.json').read_text())
session = boto3.Session(profile_name='udacity', region_name=state['region'])
assert session.client('sts').get_caller_identity()['Account'] == state['account']
client = session.client('bedrock-agentcore')
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--only', nargs='+', choices=SCENARIOS)
args = parser.parse_args()
for name in args.only or ['01-order', '02-refund', '03-rag', '05-discount', '06-browser']:
    prompt, prefix = SCENARIOS[name]
    payload = {'prompt': prompt, 'customer_id': 'CUST-123',
               'session_id': prefix + '-trace-' + uuid.uuid4().hex[:8], 'include_tool_trace': True}
    runtime_session = str(uuid.uuid4())
    print('Capturing ' + name, flush=True)
    response = client.invoke_agent_runtime(agentRuntimeArn=state['runtime_arn'], runtimeSessionId=runtime_session,
                                           payload=json.dumps(payload).encode(), contentType='application/json')
    body = json.loads(response['response'].read())
    if isinstance(body, str):
        body = json.loads(body)
    record = {'utc': datetime.now(timezone.utc).isoformat(), 'runtime_arn': state['runtime_arn'],
              'runtime_session': runtime_session, 'request_id': response['ResponseMetadata']['RequestId'],
              'payload': payload, 'response': body}
    path = root / 'evidence' / (name + '-tool-trace.json')
    path.write_text(json.dumps(record, indent=2, default=str) + '\n')
    assert isinstance(body, dict) and body.get('response') and body.get('tool_trace'), record
    uses = {b['toolUse']['toolUseId']: b['toolUse'] for b in body['tool_trace'] if 'toolUse' in b}
    results = {b['toolResult']['toolUseId']: b['toolResult'] for b in body['tool_trace'] if 'toolResult' in b}
    for key, tool in uses.items():
        result = results[key]
        assert result.get('content'), result
        if tool['name'] != 'browser':
            assert result.get('status') == 'success' and not result.get('isError'), result
        print(tool['name'] + ': ' + result['status'], flush=True)
    if name == '06-browser':
        assert any(r.get('status') == 'success' and 'Evaluation result: Learn the Latest Tech Skills' in json.dumps(r['content']) for r in results.values()), body
    if name == '05-discount':
        calc_id = next(key for key, use in uses.items() if use['name'] == 'calculate_loyalty_discount')
        calc = json.loads(results[calc_id]['content'][0]['text'])
        assert calc['calculation_source'] == 'agentcore_code_interpreter', calc
        assert {k: calc[k] for k in ['points_redeemed', 'tier_discount_pct', 'final_total', 'remaining_points']} == {
            'points_redeemed': 4000, 'tier_discount_pct': 10, 'final_total': '99.00', 'remaining_points': 250}, calc
        assert 'Final total: $99.00' in body['response'], body['response']
    print(body['response'], flush=True)

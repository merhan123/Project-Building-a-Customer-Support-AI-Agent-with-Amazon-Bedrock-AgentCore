"""Test the deployed artifact with a dummy Gateway in a temporary runtime."""
import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
import boto3

root = Path(__file__).resolve().parent
state = json.loads((root / 'resources.json').read_text())
session = boto3.Session(profile_name='udacity', region_name=state['region'])
assert session.client('sts').get_caller_identity()['Account'] == state['account']
control = session.client('bedrock-agentcore-control')
runtime = session.client('bedrock-agentcore')
source_id = state['runtime_arn'].split('/')[-1]
source = control.get_agent_runtime(agentRuntimeId=source_id)
assert source['status'] == 'READY', source['status']
fields = ['agentRuntimeArtifact', 'roleArn', 'networkConfiguration',
          'protocolConfiguration', 'lifecycleConfiguration']
args = {key: source[key] for key in fields if key in source}
args['agentRuntimeName'] = 'gateway_failure_probe_' + uuid.uuid4().hex[:8]
args['environmentVariables'] = dict(source.get('environmentVariables', {}),
                                    GATEWAY_URL='http://127.0.0.1:1/mcp')
created = control.create_agent_runtime(**args)
probe_id = created['agentRuntimeId']
print('Created temporary Gateway failure probe', flush=True)
try:
    for _ in range(120):
        probe = control.get_agent_runtime(agentRuntimeId=probe_id)
        if probe['status'] == 'READY':
            break
        if probe['status'] in {'CREATE_FAILED', 'UPDATE_FAILED'}:
            raise RuntimeError('Probe provisioning failed')
        time.sleep(5)
    else:
        raise TimeoutError('Probe readiness timed out')
    payload = {'prompt': 'Can you track order ORD-001?', 'customer_id': 'CUST-123',
               'session_id': 'failure-' + uuid.uuid4().hex[:8]}
    response = runtime.invoke_agent_runtime(agentRuntimeArn=created['agentRuntimeArn'],
        runtimeSessionId=str(uuid.uuid4()), payload=json.dumps(payload).encode(), contentType='application/json')
    body = json.loads(response['response'].read())
    if isinstance(body, str):
        body = json.loads(body)
    assert isinstance(body, dict) and 'Gateway' in body.get('error', ''), body
    assert 'try again later' in body['error'], body
    assert '127.0.0.1' not in json.dumps(body), body
    evidence = {'utc': datetime.now(timezone.utc).isoformat(),
                'test': 'Real deployed artifact with unreachable loopback Gateway endpoint',
                'source_runtime_version': source['agentRuntimeVersion'],
                'probe_runtime_arn': created['agentRuntimeArn'],
                'request_id': response['ResponseMetadata']['RequestId'],
                'response': body, 'passed': True}
    (root / 'evidence/gateway-failure-deployed.json').write_text(json.dumps(evidence, indent=2) + '\n')
    print(json.dumps(evidence, indent=2), flush=True)
finally:
    control.delete_agent_runtime(agentRuntimeId=probe_id)
    print('Temporary runtime deletion requested', flush=True)

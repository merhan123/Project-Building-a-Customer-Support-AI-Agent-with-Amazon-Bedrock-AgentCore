"""Capture real deployed agentcore invoke output for the six course scenarios."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import subprocess
import time
import uuid

ROOT = Path(__file__).resolve().parent
SCENARIOS = {
    '01-order': ('Can you track order ORD-001?', 't1'),
    '02-refund': ('I want to return my Kindle Paperwhite (ORD-002). Please initiate a refund.', 't2'),
    '03-rag': ('What are the benefits of the Platinum loyalty tier?', 't3'),
    '04a-memory': ('Hi, I am Jane. I prefer concise responses.', 's-A'),
    '04b-memory': ('Do you remember my name and communication preference?', 's-B'),
    '05-discount': ('I am a Gold member with 4250 points. Calculate my discount on a $150 standard order.', 't5'),
    '06-browser': ('Go to https://www.udacity.com and tell me the page title.', 't6'),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--only', nargs='+', choices=SCENARIOS)
    args = parser.parse_args()
    env = dict(os.environ, AWS_PROFILE='udacity', AWS_REGION='us-east-1', AGENTCORE_SUPPRESS_RECOMMENDATION='1', COLUMNS='160', NO_COLOR='1')
    for name in args.only or SCENARIOS:
        if name == '04b-memory' and (not args.only or '04a-memory' in args.only):
            print('Waiting 90 seconds for asynchronous memory extraction...', flush=True)
            time.sleep(90)
        prompt, session = SCENARIOS[name]
        # Fresh application sessions prevent previous test turns from satisfying recall.
        session += '-' + uuid.uuid4().hex[:8]
        payload = json.dumps({'prompt': prompt, 'customer_id': 'CUST-123', 'session_id': session, 'include_tool_trace': True})
        runtime_session = str(uuid.uuid4())
        command = [str(ROOT / 'starter/.venv/bin/agentcore'), 'invoke', payload, '--session-id', runtime_session]
        print('Running ' + name, flush=True)
        result = subprocess.run(command, cwd=ROOT / 'starter', env=env, capture_output=True, text=True, timeout=600)
        timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        output = (f'UTC: {timestamp}\nRuntime session: {runtime_session}\n'
                  f'Command: agentcore invoke {shlex.quote(payload)} --session-id {runtime_session}\n'
                  f'Exit code: {result.returncode}\n\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}')
        (ROOT / 'evidence' / f'{name}-{timestamp}.txt').write_text(output)
        print(output, flush=True)
        if result.returncode:
            raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()

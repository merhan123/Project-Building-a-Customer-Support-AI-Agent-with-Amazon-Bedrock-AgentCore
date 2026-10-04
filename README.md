# Customer Support AI Agent with Amazon Bedrock AgentCore

**Udacity — AWS AI Engineering Nanodegree**

A deployed customer support agent for a fictional e-commerce store. The agent tracks orders, processes demo refunds, answers product and policy questions, remembers customer preferences, calculates loyalty discounts, and retrieves live web content.

The implementation uses Amazon Nova 2 Lite with the Strands Agents SDK and Amazon Bedrock AgentCore. The recorded validation on **October 3, 2026** covers all six required scenarios and eight local regression tests. See the [rubric assessment](rubric-assessment.txt) for the seven required criteria and their evidence.

**Submission:** [submission.zip](submission.zip) · [Evidence index](submission/EVIDENCE.txt) · [Reflection](reflection.txt)

## Architecture

```text
agentcore invoke / InvokeAgentRuntime
                  |
         AgentCore Runtime
       Strands + Nova 2 Lite
                  |
    +-------------+-------------------+------------------+
    |             |                   |                  |
 MCP Gateway   Knowledge Base    AgentCore Memory    AgentCore tools
    |          Retrieve API      facts/preferences   Code Interpreter
    |             |                                  Browser
    |          S3 catalog
    +-- API Gateway target --> order-tracker Lambda
    +-- Lambda target -------> refund-processor Lambda
```

The module-level `BedrockAgentCoreApp` exposes an async `invoke` entrypoint and runs through `app.run()`. The model uses the inference profile `global.amazon.nova-2-lite-v1:0`.

| Capability | Implementation |
| --- | --- |
| External tools | `MCPClient` loads Gateway tools and adds them to the agent. |
| RAG | `search_knowledge_base` calls Retrieve, joins text chunks, and handles an unconfigured knowledge base. |
| Memory | `MemoryHook` retrieves strategy-tagged context and saves the original query and final response. Namespace discovery supports both `namespaceTemplates` and legacy `namespaces`. |
| Calculation | `calculate_loyalty_discount` executes self-contained Python with `Decimal` in AgentCore Code Interpreter, using `clearContext=True`. |
| Browsing | `AgentCoreBrowser` retrieves live content and closes its browser resources after invocation. |
| Tool evidence | Optional `include_tool_trace` output preserves actual tool inputs and results. |

### Gateway targets

| Target | Target type | Tools |
| --- | --- | --- |
| `order-tracker` | API Gateway | `get_order`, `get_customer_orders`, `get_customer` |
| `refund-processor` | Lambda | `initiate_refund`, `check_refund_status`, `get_return_label` |

Gateway tool names include their target prefix, for example `order-tracker___get_order`. Target names use hyphens because AWS rejected underscores during setup. The API Gateway routes use IAM authorization; the lab MCP Gateway uses `NONE` inbound authorization.

The knowledge base uses the managed S3 connector (`MANAGED_KNOWLEDGE_BASE_CONNECTOR`), backed by [product_catalog.txt](starter/product_catalog.txt). Memory uses semantic and user-preference strategies with `cs_agent/{actorId}/facts` and `cs_agent/{actorId}/preferences` namespaces.

## Project TODOs with implemented solutions

The original learning tasks are retained below as completed TODOs. Each solution points to the working implementation in [starter/main.py](starter/main.py); these are completed requirements, not unfinished code placeholders.

### TODO 1 — Configuration and initialisation ✅

**Task:** Fill in the resource IDs and initialise the runtime, model, and AWS clients.

**Solution:** Configure `GATEWAY_URL`, `KB_ID`, `MEMORY_ID`, and `REGION`. Create `app = BedrockAgentCoreApp()` at module level, initialise `BedrockModel` with `global.amazon.nova-2-lite-v1:0`, and create `MemoryClient` and the `bedrock-agent-runtime` boto3 client. The configured identifiers match the project inventory in [resources.json](resources.json).

### TODO 2 — Knowledge Base tool ✅

**Task:** Implement `search_knowledge_base(query)` using the Retrieve API and join retrieved text chunks.

**Solution:** The `@tool` function documents when to search product and policy information, checks for a missing `KB_ID`, calls `_bedrock_runtime.retrieve(...)`, extracts text from `retrievalResults`, and joins nonempty chunks with `"\n---\n"`. Empty results and service failures return descriptive messages instead of invented information.

**Validation:** The RAG test retrieves Platinum membership benefits from the catalog. See Test 3 in the recorded results below.

### TODO 3 — Long-term memory hook ✅

**Task:** Discover memory namespaces, retrieve customer context before a response, and persist the completed interaction afterward.

**Solution:** `get_namespaces` maps strategy types to `namespaceTemplates`, falling back to legacy `namespaces`. `MemoryHook` extends `HookProvider`; `register_hooks` connects `MessageAddedEvent` to `retrieve_customer_context` and `AfterInvocationEvent` to `save_support_interaction`. Retrieval searches every configured strategy namespace, tags each memory by strategy, and prepends it to the user message. Persistence extracts the original user query and final assistant response and calls `memory_client.create_event()`.

**Validation:** Two separate sessions with the same customer ID demonstrate recall of Jane and her preference for concise responses. The test runner allows 90 seconds for asynchronous extraction.

### TODO 4 — Loyalty discount tool with Code Interpreter ✅

**Task:** Implement `calculate_loyalty_discount(loyalty_points, tier, order_total, product_category)` with sandboxed business rules and an unavailable-interpreter fallback.

**Solution:** The `@tool` function builds self-contained Python using `Decimal` for points redemption, tier discounts, and category earn rates. It executes the code through `code_session(REGION).invoke("executeCode", ...)` with `clearContext=True`, parses and validates the result, and returns JSON containing `points_redeemed`, `tier_discount_pct`, `final_total`, and `remaining_points`. A labeled fallback computes only the tier discount. `ground_calculation_response` keeps the displayed and saved amounts consistent with the actual tool result.

**Validation:** The Gold-member example with 4,250 points and a $150 purchase returns a $99 final total. Local tests also exercise rounding, invalid inputs, and fallback behavior.

### TODO 5 — Main entrypoint and tool integration ✅

**Task:** Implement `invoke(payload, context)`, connect Gateway tools, attach memory and browser capabilities, and return the agent response.

**Solution:** The async function uses `@app.entrypoint`, validates the prompt and customer/session IDs, creates `MemoryHook` and `AgentCoreBrowser(region=REGION, ...)`, and restores session history. Inside an `MCPClient` connection, it loads all Gateway tool pages and combines them with `search_knowledge_base`, `calculate_loyalty_discount`, and `browser.browser`. The Strands agent runs with these tools and the memory hook. The entrypoint returns the final response, optionally includes real tool traces, and closes browser resources in `finally`. The module launches with `app.run()`.

**Validation:** Order and refund traces show successful API-backed and Lambda-backed calls; the browser trace demonstrates live page retrieval.

### TODO 6 — Deploy to AgentCore and verify ✅

**Task:** Configure and deploy the cloud runtime, grant integration permissions, and invoke the deployed agent.

**Solution:** Use the Python Starter Toolkit's `agentcore configure` and `agentcore deploy`, then run `setup_permissions.py --profile udacity` to grant access to the configured integrations. The exact commands are retained in **Provisioning and deployment** below. Toolkit automatic memory creation is disabled because this implementation uses its separately configured memory resource.

**Validation:** The submission contains seven `agentcore invoke` transcripts covering all six scenarios, including both memory sessions, plus full SDK tool traces and eight passing local regression tests.

## Repository contents

```text
.
├── README.md
├── resources.json                 # Non-secret AWS resource inventory
├── provision.py                   # Resumable, account-guarded lab provisioning
├── verify_setup.py                # Live Gateway, knowledge base, and memory checks
├── run_submission_tests.py         # CLI evidence for six scenarios / seven calls
├── capture_tool_traces.py          # Complete SDK responses with tool traces
├── rubric-assessment.txt           # Criterion-by-criterion assessment
├── reflection.txt                  # 311-word project reflection
├── starter/
│   ├── main.py                    # Completed runtime agent
│   ├── setup_permissions.py       # Runtime integration permissions
│   ├── pyproject.toml
│   ├── uv.lock
│   ├── product_catalog.txt
│   └── lambda/
│       ├── order_tracker.py
│       ├── refund_processor.py
│       └── lambda_schema
├── tests/test_agent.py             # Eight local regression tests
├── evidence/                      # Live results plus diagnostic attempts
├── submission/                    # Curated submission files and evidence
└── submission.zip
```

## Gateway review revision (October 4)

Gateway setup and paginated discovery now log successful tool counts and return
safe, actionable errors for failures or empty discovery. `GATEWAY_URL` accepts
an environment override for testing. Application error logs omit raw transport
exception text to avoid exposing endpoint credentials. Ten offline tests pass,
including Gateway failure and success cases; see
[revision details](submission/REVISION.txt). Runtime version 4 was deployed and verified on October 4. Fresh order/refund
traces passed, and a temporary cloud runtime with an unreachable dummy Gateway
returned an actionable error. Temporary runtime deletion was confirmed. Other
capability evidence remains from October 3.

## Local setup

Use Python **3.13**, [uv](https://docs.astral.sh/uv/), AWS CLI v2, and a valid AWS lab session with access to the required services and Nova 2 Lite. This project uses the **Python Bedrock AgentCore Starter Toolkit CLI** installed by its dependencies. The npm AgentCore CLI has a different project format; the commands below use the Python virtual environment explicitly.

From the repository root:

```bash
cd starter
uv sync --python 3.13 --locked
cd ..
```

Configure temporary credentials in the local AWS profile named `udacity`, including the session token. Keep credentials outside the repository. Validate the session before running cloud commands:

```bash
aws sts get-caller-identity --profile udacity --region us-east-1
export AWS_PROFILE=udacity
export AWS_REGION=us-east-1
```

The current lab resources belong to account `327887916689` in `us-east-1`. [resources.json](resources.json) records their identifiers, and [main.py](starter/main.py) contains `GATEWAY_URL`, `KB_ID`, `MEMORY_ID`, and `REGION`. Provisioning is guarded against use in a different account. A different lab account requires updating the resource configuration and reviewing that guard before provisioning.

## Validate the project

Run these commands from the repository root.

### Local regression tests

```bash
starter/.venv/bin/python -m unittest discover -s tests -v
```

The tests cover discount rules and rounding, invalid inputs, sandbox fallback, incomplete sandbox output, grounding displayed and saved totals in the calculator result, memory namespace compatibility, memory persistence, and knowledge-base responses. AWS interactions are mocked in these tests.

### Live setup and submission tests

These commands call AWS and require an active lab session. Cloud resources and invocations may incur charges.

```bash
starter/.venv/bin/python verify_setup.py
starter/.venv/bin/python run_submission_tests.py
starter/.venv/bin/python capture_tool_traces.py
```

`run_submission_tests.py` records timestamped `agentcore invoke` terminal output. It runs both memory sessions with the same customer ID and separate application/runtime sessions, waiting 90 seconds between them for asynchronous memory extraction. Extraction latency varies; the recorded successful run required about 64 seconds, so an earlier 45-second attempt was too soon.

`capture_tool_traces.py` invokes the same runtime through the AWS SDK with `include_tool_trace=true`. The Starter Toolkit CLI displays only the response text and omits sibling trace fields, so these JSON files supplement the CLI transcripts with complete tool inputs and results. Its default scenarios are order, refund, RAG, discount, and browser.

To rerun one scenario:

```bash
starter/.venv/bin/python run_submission_tests.py --only 05-discount
starter/.venv/bin/python capture_tool_traces.py --only 05-discount
```

Available scenario names are `01-order`, `02-refund`, `03-rag`, `04a-memory`, `04b-memory`, `05-discount`, and `06-browser`. Inspect response content as well as exit status: a successful CLI command alone does not prove that every tool succeeded.

### Manual invocation

From `starter/`, with the AWS profile and region exported as above:

```bash
.venv/bin/agentcore invoke '{"prompt":"Where is my order ORD-001?","customer_id":"CUST-123","session_id":"manual-order-001"}'
```

## Recorded results

These are saved results from October 3, 2026, rather than a guarantee that a temporary lab environment remains available.

| Test | Observed result | Submission evidence |
| --- | --- | --- |
| 1 — Order tracking | `ORD-001` shipped via UPS, tracking `TRK987654321`; API-backed tool returned order details. | [CLI output](submission/evidence/01-order-20261003T163816Z.txt), [tool trace](submission/evidence/01-order-tool-trace.json) |
| 2 — Refund | Refund approved for `$139.99`, with a 3–5 business-day estimate; Lambda-backed tool returned a refund record. | [CLI output](submission/evidence/02-refund-20261003T163826Z.txt), [tool trace](submission/evidence/02-refund-tool-trace.json) |
| 3 — RAG | Retrieved Platinum benefits: free same-day shipping, 15% discount, and priority support. | [CLI output](submission/evidence/03-rag-20261003T163837Z.txt), [tool trace](submission/evidence/03-rag-tool-trace.json) |
| 4 — Memory | Recalled Jane and her preference for concise responses across separate sessions for `CUST-123`. | [Session A](submission/evidence/04a-memory-20261003T162808Z.txt), [Session B](submission/evidence/04b-memory-20261003T163141Z.txt) |
| 5 — Discount | Redeemed 4,000 points, applied a 10% tier discount, and returned a final total of `$99.00`. | [CLI output](submission/evidence/05-discount-20261003T172300Z.txt), [tool trace](submission/evidence/05-discount-tool-trace.json) |
| 6 — Browser | Navigated to Udacity and retrieved the live page title: “Learn the Latest Tech Skills; Advance Your Career \| Udacity”. | [CLI output](submission/evidence/06-browser-20261003T164033Z.txt), [tool trace](submission/evidence/06-browser-tool-trace.json) |

The browser trace retains three invalid session-name attempts before successful recovery, navigation, HTML retrieval, and title evaluation. CLI and SDK traces are separate live conversations, so generated refund IDs can differ. Earlier failed attempts remain in the root `evidence/` directory; the submission includes the selected successful results and transparent trace history.

The [local test log](submission/evidence/local-tests.txt) records eight passing tests. Its intentional sandbox-unavailable traceback exercises the fallback path.

## Loyalty calculation behavior

Points redeem at 100 points per dollar, in blocks of 500, capped at 50% of the original purchase total. Tier discounts apply after points redemption: Silver 0%, Gold 10%, and Platinum 15%. Earn rates are 1 point per final dollar for standard purchases, 2 for devices, and 5 for fresh purchases.

For 4,250 points, Gold membership, and a `$150.00` standard purchase:

| Output | Value |
| --- | --- |
| `points_redeemed` | `4000` (`$40.00`) |
| `tier_discount_pct` | `10` (`$11.00` after redemption) |
| `final_total` | `99.00` |
| `remaining_points` | `250` before new earnings |
| New points earned | `99` |
| Balance including earnings | `349` |

The tool validates the sandbox result and returns structured JSON. If the interpreter is unavailable or returns incomplete data, a labeled fallback calculates only the tier discount and does not claim points redemption or new earnings.

The displayed response and saved memory are rendered directly from the calculator result. This fixes an observed case where the model stated `$95.00` despite the sandbox correctly returning `$99.00`.

## Provisioning and deployment

The existing resource inventory supports resuming lab setup:

```bash
# Repository root; review resources.json and the account guard first.
starter/.venv/bin/python provision.py
```

For an existing configured deployment, run from `starter/`:

```bash
PATH="$PWD/../.tools/bin:$PATH" UV_CACHE_DIR="$PWD/../.uv-cache" \
AWS_PROFILE=udacity AWS_REGION=us-east-1 AGENTCORE_SUPPRESS_RECOMMENDATION=1 \
.venv/bin/agentcore deploy
```

The PATH prefix supports this workspace's local `uv` installation; a normal `uv` installation on PATH also works.

If local `.bedrock_agentcore.yaml` configuration is missing, configure the runtime first from `starter/`:

```bash
PATH="$PWD/../.tools/bin:$PATH" \
AWS_PROFILE=udacity AWS_REGION=us-east-1 \
.venv/bin/agentcore configure \
  --entrypoint main.py \
  --name customer_support_agent \
  --deployment-type direct_code_deploy \
  --runtime PYTHON_3_13 \
  --disable-memory \
  --region us-east-1 \
  --idle-timeout 60 \
  --max-lifetime 1800 \
  --non-interactive
```

`--disable-memory` disables Toolkit-created memory; the agent still uses the explicitly configured project memory resource. After deploying, attach the runtime integration permissions:

```bash
.venv/bin/python setup_permissions.py --profile udacity
```

Rerun live validation after code or infrastructure changes. Runtime logging and X-Ray tracing were enabled for the recorded deployment.

## Submission and production considerations

[submission.zip](submission.zip) contains the completed `main.py`, dependency files, seven CLI transcripts, five full tool traces, setup and local test evidence, the rubric assessment, and the 311-word reflection. Review the reflection before submitting it as your own account of the work.

Required criteria are mapped in [rubric-assessment.txt](rubric-assessment.txt). Pydantic response models, long-conversation summarization, and a different business domain are optional enhancements and are not claimed by this implementation.

This is a lab configuration. For production, authenticate the MCP Gateway, derive customer identity from a trusted login instead of caller-supplied `customer_id`, and enforce authorization before accessing orders or issuing refunds. Replace the demo refund implementation with durable, idempotent processing. Restrict and redact debug tool traces, set memory retention policies, and monitor latency, failures, and service costs.

After grading, remove unneeded project resources using [resources.json](resources.json) as the inventory: runtime, Gateway targets and Gateway, memory, knowledge base and data source, API, Lambda functions, project S3 objects, and dedicated roles/log groups. Shared deployment buckets may contain other artifacts; remove only this project's objects unless the bucket is confirmed dedicated and empty. Expired credentials do not delete deployed resources.

## References

- [Amazon Bedrock AgentCore documentation](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/what-is-bedrock-agentcore.html)
- [AgentCore Starter Toolkit](https://github.com/aws/bedrock-agentcore-starter-toolkit)
- [Strands Agents documentation](https://strandsagents.com/latest/)
- [Amazon Bedrock Knowledge Bases](https://docs.aws.amazon.com/bedrock/latest/userguide/knowledge-base.html)

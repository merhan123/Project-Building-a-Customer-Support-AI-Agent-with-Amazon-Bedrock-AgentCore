"""Provision the course lab. Run with starter/.venv/bin/python provision.py.

Records created resource IDs for resuming setup and later cleanup. Never stores
credentials. Uses only the explicitly named udacity AWS profile.
"""
import io
import json
import time
import zipfile
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

ROOT = Path(__file__).resolve().parent
STATE_FILE = ROOT / "resources.json"
state = json.loads(STATE_FILE.read_text()) if STATE_FILE.exists() else {}
session = boto3.Session(profile_name="udacity", region_name="us-east-1")
account = session.client("sts").get_caller_identity()["Account"]
if account != "327887916689":
    raise RuntimeError("Refusing to provision outside the verified Udacity account")
iam = session.client("iam")
control = session.client("bedrock-agentcore-control")
bedrock = session.client("bedrock-agent")


def save(**values):
    state.update(values)
    STATE_FILE.write_text(json.dumps(state, indent=2) + "\n")
    print(json.dumps(values), flush=True)


def role(name, service, statements):
    trust = {"Version": "2012-10-17", "Statement": [{"Effect": "Allow",
        "Principal": {"Service": service}, "Action": "sts:AssumeRole",
        "Condition": {"StringEquals": {"aws:SourceAccount": account}}}]}
    try:
        arn = iam.get_role(RoleName=name)["Role"]["Arn"]
    except iam.exceptions.NoSuchEntityException:
        arn = iam.create_role(RoleName=name, AssumeRolePolicyDocument=json.dumps(trust))["Role"]["Arn"]
        save(**{name: arn})
    iam.put_role_policy(RoleName=name, PolicyName="CustomerSupportProject",
        PolicyDocument=json.dumps({"Version": "2012-10-17", "Statement": statements}))
    return arn


def wait_ready(get, desired, field="status", limit=120):
    for _ in range(limit):
        value = get()
        status = value.get(field)
        if status in desired:
            return value
        if status in {"FAILED", "CREATE_FAILED", "UPDATE_FAILED"}:
            raise RuntimeError(json.dumps(value, default=str))
        print("Waiting:", status, flush=True)
        time.sleep(5)
    raise TimeoutError("Resource did not become ready; rerun to resume")


def main():
    save(account=account, region="us-east-1")
    lambda_role = role("CustomerSupportLambdaRole", "lambda.amazonaws.com", [
        {"Effect": "Allow", "Action": "logs:CreateLogGroup", "Resource": f"arn:aws:logs:us-east-1:{account}:*"},
        {"Effect": "Allow", "Action": ["logs:CreateLogStream", "logs:PutLogEvents"],
         "Resource": f"arn:aws:logs:us-east-1:{account}:log-group:/aws/lambda/*:*"}])
    lam = session.client("lambda")
    for name, module in [("order-tracker", "order_tracker"), ("refund-processor", "refund_processor")]:
        try:
            fn = lam.get_function(FunctionName=name)["Configuration"]
        except lam.exceptions.ResourceNotFoundException:
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as archive:
                archive.write(ROOT / "starter/lambda" / (module + ".py"), module + ".py")
            for attempt in range(12):
                try:
                    fn = lam.create_function(FunctionName=name, Runtime="python3.12", Role=lambda_role,
                        Handler=module + ".lambda_handler", Code={"ZipFile": buf.getvalue()}, Timeout=30)
                    break
                except lam.exceptions.InvalidParameterValueException as error:
                    if "cannot be assumed" not in str(error) or attempt == 11:
                        raise
                    time.sleep(5)
        save(**{module + "_arn": fn["FunctionArn"]})
        lam.get_waiter("function_active_v2").wait(FunctionName=name)

    api = session.client("apigateway")
    if "rest_api_id" not in state:
        existing = [x for x in api.get_rest_apis(limit=500)["items"] if x["name"] == "CustomerSupportAPI"]
        result = existing[0] if existing else api.create_rest_api(name="CustomerSupportAPI", endpointConfiguration={"types": ["REGIONAL"]})
        save(rest_api_id=result["id"], api_stage="prod")
    api_id = state["rest_api_id"]
    resources = {x["path"]: x for x in api.get_resources(restApiId=api_id, limit=500)["items"]}
    for path in ["/orders", "/orders/{order_id}", "/customers", "/customers/{customer_id}", "/customers/{customer_id}/orders"]:
        if path not in resources:
            parent, part = path.rsplit("/", 1)
            resources[path] = api.create_resource(restApiId=api_id, parentId=resources[parent or "/"]["id"], pathPart=part)
    routes = [("/orders/{order_id}", "get_order"), ("/customers/{customer_id}/orders", "get_customer_orders"), ("/customers/{customer_id}", "get_customer")]
    for path, operation in routes:
        resource_id = resources[path]["id"]
        if "GET" not in resources[path].get("resourceMethods", {}):
            api.put_method(restApiId=api_id, resourceId=resource_id, httpMethod="GET", authorizationType="AWS_IAM",
                operationName=operation, requestParameters={"method.request.path." + ("order_id" if "{order_id}" in path else "customer_id"): True})
        api.put_integration(restApiId=api_id, resourceId=resource_id, httpMethod="GET", type="AWS_PROXY", integrationHttpMethod="POST",
            uri=f"arn:aws:apigateway:us-east-1:lambda:path/2015-03-31/functions/{state['order_tracker_arn']}/invocations")
        method = api.get_method(restApiId=api_id, resourceId=resource_id, httpMethod="GET")
        if "200" not in method.get("methodResponses", {}):
            api.put_method_response(restApiId=api_id, resourceId=resource_id, httpMethod="GET", statusCode="200", responseModels={"application/json": "Empty"})
    try:
        lam.add_permission(FunctionName="order-tracker", StatementId="CustomerSupportAPI", Action="lambda:InvokeFunction",
            Principal="apigateway.amazonaws.com", SourceArn=f"arn:aws:execute-api:us-east-1:{account}:{api_id}/*/GET/*")
    except lam.exceptions.ResourceConflictException:
        pass
    api.create_deployment(restApiId=api_id, stageName=state["api_stage"])

    gateway_role = role("CustomerSupportGatewayRole", "bedrock-agentcore.amazonaws.com", [
        {"Effect": "Allow", "Action": "lambda:InvokeFunction", "Resource": state["refund_processor_arn"]},
        {"Effect": "Allow", "Action": "execute-api:Invoke", "Resource": f"arn:aws:execute-api:us-east-1:{account}:{api_id}/prod/GET/*"},
        {"Effect": "Allow", "Action": "apigateway:GET", "Resource": [f"arn:aws:apigateway:us-east-1::/restapis/{api_id}", f"arn:aws:apigateway:us-east-1::/restapis/{api_id}/*"]}])
    if "gateway_id" not in state:
        g = control.create_gateway(name="CustomerSupportGateway", roleArn=gateway_role, protocolType="MCP", authorizerType="NONE")
        save(gateway_id=g["gatewayId"], gateway_url=g["gatewayUrl"])
    wait_ready(lambda: control.get_gateway(gatewayIdentifier=state["gateway_id"]), {"READY"})
    targets = {x["name"]: x for x in control.list_gateway_targets(gatewayIdentifier=state["gateway_id"])["items"]}
    configs = {
        "order_tracker": {"apiGateway": {"restApiId": api_id, "stage": "prod", "apiGatewayToolConfiguration": {
            "toolOverrides": [{"name": name, "path": path, "method": "GET", "description": name.replace("_", " ")} for path, name in routes],
            "toolFilters": [{"filterPath": path, "methods": ["GET"]} for path, _ in routes]}}},
        "refund_processor": {"lambda": {"lambdaArn": state["refund_processor_arn"], "toolSchema": {"inlinePayload": json.loads((ROOT / "starter/lambda/lambda_schema").read_text())}}}}
    for name, config in configs.items():
        target_name = name.replace("_", "-")
        t = targets.get(target_name) or control.create_gateway_target(gatewayIdentifier=state["gateway_id"], name=target_name,
            targetConfiguration={"mcp": config}, credentialProviderConfigurations=[{"credentialProviderType": "GATEWAY_IAM_ROLE"}])
        save(**{name + "_target_id": t["targetId"]})
        wait_ready(lambda: control.get_gateway_target(gatewayIdentifier=state["gateway_id"], targetId=t["targetId"]), {"READY"})

    s3 = session.client("s3")
    bucket = f"customer-support-catalog-{account}-us-east-1"
    if "catalog_bucket" not in state:
        s3.create_bucket(Bucket=bucket)
        save(catalog_bucket=bucket)
        s3.put_public_access_block(Bucket=bucket, PublicAccessBlockConfiguration={k: True for k in ["BlockPublicAcls", "IgnorePublicAcls", "BlockPublicPolicy", "RestrictPublicBuckets"]})
    s3.upload_file(str(ROOT / "starter/product_catalog.txt"), bucket, "product_catalog.txt")
    kb_role = role("CustomerSupportKnowledgeBaseRole", "bedrock.amazonaws.com", [
        {"Effect": "Allow", "Action": "s3:ListBucket", "Resource": f"arn:aws:s3:::{bucket}"},
        {"Effect": "Allow", "Action": "s3:GetObject", "Resource": f"arn:aws:s3:::{bucket}/product_catalog.txt"}])
    if "kb_id" not in state:
        kb = bedrock.create_knowledge_base(name="CustomerSupportKB", roleArn=kb_role,
            knowledgeBaseConfiguration={"type": "MANAGED", "managedKnowledgeBaseConfiguration": {"embeddingModelType": "MANAGED"}})["knowledgeBase"]
        save(kb_id=kb["knowledgeBaseId"])
    wait_ready(lambda: bedrock.get_knowledge_base(knowledgeBaseId=state["kb_id"])["knowledgeBase"], {"ACTIVE"})
    if "data_source_id" not in state:
        ds = bedrock.create_data_source(knowledgeBaseId=state["kb_id"], name="ProductCatalog", dataDeletionPolicy="DELETE",
            dataSourceConfiguration={"type": "MANAGED_KNOWLEDGE_BASE_CONNECTOR", "managedKnowledgeBaseConnectorConfiguration": {
                "connectorParameters": {"type": "S3", "version": "1", "connectionConfiguration": {
                    "bucketName": bucket, "bucketOwnerAccountId": account}}}})["dataSource"]
        save(data_source_id=ds["dataSourceId"])
    wait_ready(lambda: bedrock.get_data_source(knowledgeBaseId=state["kb_id"], dataSourceId=state["data_source_id"])["dataSource"], {"AVAILABLE"})
    if "ingestion_job_id" not in state:
        job = bedrock.start_ingestion_job(knowledgeBaseId=state["kb_id"], dataSourceId=state["data_source_id"])["ingestionJob"]
        save(ingestion_job_id=job["ingestionJobId"])
    wait_ready(lambda: bedrock.get_ingestion_job(knowledgeBaseId=state["kb_id"], dataSourceId=state["data_source_id"], ingestionJobId=state["ingestion_job_id"])["ingestionJob"], {"COMPLETE"})
    if "memory_id" not in state:
        memory = control.create_memory(name="CustomerSupportMemory", eventExpiryDuration=30, memoryStrategies=[
            {"semanticMemoryStrategy": {"name": "customer_facts", "namespaces": ["cs_agent/{actorId}/facts"]}},
            {"userPreferenceMemoryStrategy": {"name": "customer_preferences", "namespaces": ["cs_agent/{actorId}/preferences"]}}])["memory"]
        save(memory_id=memory["id"])
    wait_ready(lambda: control.get_memory(memoryId=state["memory_id"])["memory"], {"ACTIVE"})
    print("Infrastructure ready.", flush=True)


if __name__ == "__main__":
    main()

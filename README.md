# Observability SDK for Gen AI Applications

Enables developers to instrument Generative AI applications with comprehensive tracing, evaluations, and logging—without exposing sensitive data. Supports faster debugging, performance optimization, and compliance in production deployments.

## Purpose
This SDK provides observability for Gen AI workflows, including:
- **Tracing**: Hierarchical spans for pre/post-processing, tools, and LLM calls.
- **Evaluations**: Quality (QA, Hallucination) and safety (PII, Toxicity) checks.
- **Logging**: Secure, anonymized data export to Phoenix or Arize.

## Key Benefits
- **Observability**: Captures complex workflows (e.g., RAG, agents, tools) with metrics like latency and token usage.
- **Evaluations**: Online (sampled, real-time) and offline (batch) evals; anonymizes data before logging.
- **Privacy & Compliance**: Uses Presidio for NER-based PII detection/redaction, aligning with GDPR/CCPA/HIPAA.
- **Efficiency**: Singleton design for minimal overhead; lightweight decorators/context managers.
- **Cost Savings**: Reduces downtime via alerts; optimizes LLM calls through custom gateways.
- **ROI**: Quick adoption via pip installs + env config; measurable via reduced incidents and audits.

## Target Users
Developers and teams building Gen AI tools. Supports frameworks like ROMA, LangChain via tracing wrappers.

## Dependencies
- Python 3.12+
- OpenAI SDK
- Phoenix/Arize
- Presidio (for PII)
- rouge-score
- Other: httpx, requests, pandas, etc.
- No internet required for PII detection (offline NER).

## Modes
- **Local (Phoenix)**: For dev/testing with UI-based debugging.
- **AX Enterprise (Arize)**: Cloud production with secure logging.

## Limitations
- PII detection is probabilistic (not 100%—combine with reviews).
- Evals rely on LLM accuracy.
- Customize for specific workflows.

## Architecture Overview
- **Singleton Pattern**: One instance per app to avoid redundant setups.
- **Tracing**: OpenTelemetry-based spans/attributes; exports to Arize dashboards.
- **Evaluations**: Phoenix evals library + custom PII via Presidio.
  - Online: 10% sampling for QA/Hallucination; full for others.
  - Offline: Batch all traces.
- **Anonymization**: Presidio scans/anonymizes before logging.
- **Custom LLM Routing**: Eval calls route through company gateway with Apigee keys/headers.

## Components
- **Tracer**: OTEL-based for workflow spans.
- **Clients**: Phoenix (local) or Arize (AX) for logging.
- **Evaluators**: QA, Hallucination, Relevance, Toxicity, PII (Presidio-enhanced), ROUGE.
- **Wrappers**: `@workflow`, `@tool_span`, `trace_block` for easy instrumentation.

## Key Features in Detail

### Tracing and Monitoring
- Hierarchical spans: Pre-process (e.g., RAG), tools, LLM calls, post-process.
- Attributes: Custom annotations (prompt/output, tokens, errors).
- Usage: Decorate functions or use context managers; auto-exports.

#### Using trace_block with Custom Attributes
Pass a dictionary of attributes to be logged in Arize spans:
```python
# Basic trace_block with attributes
custom_attrs = {
    "operation.type": "data_processing",
    "user.id": "user123",
    "request.id": "req456",
    "model.version": "v1.2",
    "custom.metric": 42.5
}

with sdk.trace_block("data_processing_step", attributes=custom_attrs):
    # Your code here - attributes logged in Arize
    pass

# Nested blocks with inherited attributes
with sdk.trace_block("workflow", attributes={"workflow.name": "example"}):
    with sdk.trace_block("step1", attributes={"step": 1}):
        # Inherits parent attributes + adds own
        pass
```

### Evaluations
- **Online**: Sampled (10%) for QA/Hallucination; real-time logging. In Phoenix, adds span annotations.
- **Offline**: Batch all traces; full evals.
- **PII Handling**: Presidio NER detects/anonymizes (avoids logging emails/SSNs).
- **Other Evals**: LLM-based (QA: correctness; Hallucination: factuality) via custom gateway; ROUGE for similarity.

### Privacy Integration (Presidio)
- Offline NER: Detects entities without cloud calls.
- Anonymization: Replaces PII with `[REDACTED]`.
- Customizable: Add entity types via recognizers.

## Configuration
Set via environment variables:
- `USE_AX_MODE`: `true` for AX, `false` for Phoenix.
- `CUSTOM_LLM_URL`: Gateway URL.
- `APIGEE_KEY`: API key.
- `CUSTOM_HEADERS_JSON`: JSON headers.
- `USE_CASE_ID`: Use case identifier sent in headers (default: 'default_use_case').
- Other: Endpoints, sampling rates, etc.

### Headers for Online Sampling
For online evaluations, the SDK automatically includes:
- `Authorization: Bearer <token>` - Apigee access token
- `X-Use-Case-ID: <use_case_id>` - From USE_CASE_ID env var
- `X-Request-ID: <uuid>` - Auto-generated UUID for each request

These headers are refreshed automatically and used for all LLM evaluation calls.

## Deployment and Scaling
- Lightweight: Minimal runtime impact.
- Singleton: Thread-safe for multi-class use.
- Integration: Works with ROMA/LangChain; extensible.

## Usage Guidelines

### Initialization
```python
from observability_sdk import ObservabilitySDK
sdk = ObservabilitySDK()  # Singleton
```

### Tracing
- **Workflow**: `@sdk.workflow def my_func(...): ...`
- **Tools**: `@sdk.tool_span def my_tool(...): ...`
- **LLM**: `@sdk.llm_span def my_llm_call(...): ...`
- **Blocks**: `with sdk.trace_block("name"): ...`
- **Agent**: `@sdk.agent_span def my_agent(...): ...`

### Evaluations
- Automatic in workflows.
- Manual: `sdk.run_online_evals(span_id, input, ref, output)`
- Offline: `sdk.run_offline_evals()` at shutdown.

### Example Usage
```python
@sdk.workflow
def complex_workflow(input_data: str, reference: str = "") -> str:
    # Workflow logic
    result = "processed output"
    return result

# Run
result = complex_workflow("input", reference="ref")
sdk.shutdown()
```

#### Advanced Reasoning Agent with Complex Tracing
```python
@sdk.workflow
def advanced_reasoning_agent(user_query: str) -> str:
    """Multi-step reasoning with tool orchestration and detailed tracing."""
    
    # Step 1: Query Analysis
    with sdk.trace_block("query_analysis", attributes={
        "step": 1, "phase": "analysis", "analysis.type": "semantic_parsing"
    }):
        analysis_result, _ = reasoning_llm(f"Analyze: {user_query}")
    
    # Step 2: Multi-step Reasoning Chain
    with sdk.trace_block("reasoning_chain", attributes={
        "step": 2, "phase": "reasoning", "chain.length": 3
    }):
        # Multiple reasoning steps with different tools
        decompose_result, _ = reasoning_llm("Decompose the problem...")
        plan_result, _ = reasoning_llm("Plan tool usage...")
        validate_result, _ = reasoning_llm("Validate approach...")
    
    # Step 3: Tool Orchestration
    with sdk.trace_block("tool_orchestration", attributes={
        "step": 3, "phase": "execution", "tools.count": 2
    }):
        math_result = toolkit.math_tool("x**2 + sin(x)")
        search_result = toolkit.search_tool("AI trends")
    
    # Step 4: Synthesis
    with sdk.trace_block("synthesis", attributes={
        "step": 4, "phase": "synthesis", "confidence.threshold": 0.8
    }):
        final_result, _ = reasoning_llm(f"Synthesize: {math_result}, {search_result}")
    
    return final_result
```

#### Online Sampling with Headers
When online evaluations run, they automatically include authenticated headers:
```python
# Headers sent with each evaluation request:
{
    "Authorization": "Bearer eyJhbGciOiJSUzI1NiIs...",
    "X-Use-Case-ID": "genai_observability_prod",
    "X-Request-ID": "550e8400-e29b-41d4-a716-446655440000"  # UUID
}
```
Sampling metadata is logged in span attributes for tracking evaluation coverage.

## Installation
```bash
pip install -r requirements.txt
```

## Contributing
- Follow singleton pattern.
- Add tests for new features.
- Update spec for changes.

## License
[Add license info]
# complex_sample_app.py (More Complex Usage Example with Multiple Workflows, Real OpenAI Call, Error Handling)
import os
import time
import json
import sympy as sp
from openai import OpenAI
from observability_sdk import ObservabilitySDK

# Environment Configuration (set these in .env file or uncomment here)
# Core Configuration
# os.environ['PHOENIX_COLLECTOR_ENDPOINT'] = 'https://phoenix.example.com:6006/'  # Phoenix UI endpoint
# os.environ['PHOENIX_OTLP_ENDPOINT'] = 'https://phoenix.example.com:6006/v1/traces'  # Phoenix OTLP endpoint
# os.environ['ARIZE_ENDPOINT'] = 'https://otlp.arize.com/v1'  # Arize OTLP endpoint
# os.environ['ALLOW_INSECURE_CONNECTION'] = 'true'  # Disable SSL cert validation
# os.environ['USE_AX_MODE'] = 'false'  # 'true' for Arize Enterprise, 'false' for Phoenix
# os.environ['ONLINE_SAMPLE_RATIO'] = '0.1'  # Sampling ratio for evaluations (0.0-1.0)
# os.environ['SAMPLE_TRACKING_ENABLED'] = 'true'  # Enable sampling metadata

# Authentication and API Keys
# os.environ['OPENAI_API_KEY'] = 'your_openai_key_here'  # OpenAI API key
# os.environ['APIGEE_KEY'] = 'your_apigee_key_here'  # Custom gateway key (optional)
# os.environ['ARIZE_SPACE_KEY'] = 'your_arize_space_key'  # Arize space key
# os.environ['ARIZE_API_KEY'] = 'your_arize_api_key'  # Arize API key

# Advanced/Optional
# os.environ['CUSTOM_HEADERS_JSON'] = '{"X-Custom": "value"}'  # Custom headers for LLM requests
# os.environ['REFRESH_INTERVAL_SECONDS'] = '3600'  # Refresh interval for keys/headers
# os.environ['CUSTOM_SSL_CERT_FILE'] = '/path/to/cert.pem'  # Custom SSL cert file
# os.environ['APIGEE_ENDPOINT'] = 'https://your-apigee-endpoint/oauth/token'  # Apigee token endpoint
# os.environ['APIGEE_CLIENT_ID'] = 'your_client_id'  # Apigee client ID
# os.environ['APIGEE_CLIENT_SECRET'] = 'your_client_secret'  # Apigee client secret
# os.environ['APIGEE_REFRESH_INTERVAL'] = '3600'  # Apigee token refresh interval

# Example settings (uncomment to override)
os.environ['PHOENIX_COLLECTOR_ENDPOINT'] = 'http://localhost:6006/'  # Local Phoenix
os.environ['PHOENIX_OTLP_ENDPOINT'] = 'http://localhost:6006/v1/traces'
os.environ['ALLOW_INSECURE_CONNECTION'] = 'true'
os.environ['REFRESH_INTERVAL_SECONDS'] = '3600'
os.environ['OPENAI_API_KEY'] = 'your_openai_key_here'
os.environ['USE_AX_MODE'] = 'false'
os.environ['ONLINE_SAMPLE_RATIO'] = '0.5'

sdk = ObservabilitySDK()
openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

class ToolKit:
    """Separate class for tools, each traced with @sdk.tool_span."""
    
    def __init__(self, sdk):
        self.sdk = sdk
    
    @sdk.tool_span
    def math_tool(self, data: str) -> str:
        """Math tool using SymPy."""
        try:
            x = sp.symbols('x')
            expr = sp.sympify(data)
            return str(sp.integrate(expr, x))  # Integrate as example
        except Exception as e:
            raise ValueError(f"Math tool error: {e}")
    
    @sdk.tool_span
    def search_tool(self, query: str) -> str:
        """Simulated search tool."""
        time.sleep(1.5)
        return f"Search results for '{query}': Mock data from web."
    
    @sdk.tool_span
    def weather_tool(self, location: str) -> str:
        """Simulated weather tool."""
        time.sleep(1.0)
        return f"Weather in {location}: Sunny, 75°F."
    
    @sdk.tool_span
    def calculator_tool(self, expression: str) -> str:
        """Simple calculator tool."""
        try:
            return str(eval(expression))
        except Exception as e:
            return f"Calculation error: {e}"

# Initialize ToolKit
toolkit = ToolKit(sdk)

@sdk.llm_span
def real_llm_call(prompt: str, model: str = 'gpt-4o-mini') -> tuple:
    """Real OpenAI LLM call."""
    response = openai_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    result = response.choices[0].message.content
    token_info = {
        'model': model,
        'input_tokens': response.usage.prompt_tokens,
        'output_tokens': response.usage.completion_tokens,
        'custom_attrs': {'temperature': 0.7, 'provider': 'OpenAI'}
    }
    return result, token_info

# Tool Calling LLM Workflow Example
@sdk.llm_span
def tool_calling_llm(prompt: str, model: str = 'gpt-4o-mini') -> tuple:
    """LLM with tool calling: Uses OpenAI's function calling to invoke tools."""
    tools = [
        {
            "type": "function",
            "function": {
                "name": "math_tool",
                "description": "Perform mathematical operations using SymPy.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "data": {"type": "string", "description": "Math expression to evaluate."}
                    },
                    "required": ["data"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "search_tool",
                "description": "Search for information on the web.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query."}
                    },
                    "required": ["query"]
                }
            }
        }
    ]
    
    response = openai_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        tools=tools,
        tool_choice="auto"
    )
    
    message = response.choices[0].message
    result = message.content or ""
    tool_calls = message.tool_calls or []
    
    # Execute tool calls
    for tool_call in tool_calls:
        if tool_call.function.name == "math_tool":
            args = json.loads(tool_call.function.arguments)
            tool_result = toolkit.math_tool(args["data"])
            result += f" | Math Tool: {tool_result}"
        elif tool_call.function.name == "search_tool":
            args = json.loads(tool_call.function.arguments)
            tool_result = toolkit.search_tool(args["query"])
            result += f" | Search Tool: {tool_result}"
    
    token_info = {
        'model': model,
        'input_tokens': response.usage.prompt_tokens,
        'output_tokens': response.usage.completion_tokens,
        'custom_attrs': {'tools_used': len(tool_calls), 'provider': 'OpenAI'}
    }
    return result, token_info

# Reasoning LLM Example
@sdk.llm_span
def reasoning_llm(prompt: str, model: str = 'gpt-4o-mini') -> tuple:
    """LLM with step-by-step reasoning."""
    reasoning_prompt = f"{prompt}\n\nPlease reason step-by-step and provide a final answer."
    response = openai_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": reasoning_prompt}],
        temperature=0.3  # Lower for more deterministic reasoning
    )
    result = response.choices[0].message.content
    token_info = {
        'model': model,
        'input_tokens': response.usage.prompt_tokens,
        'output_tokens': response.usage.completion_tokens,
        'custom_attrs': {'reasoning': True, 'temperature': 0.3, 'provider': 'OpenAI'}
    }
    return result, token_info

@sdk.agent_span
def advanced_agent(input_data: str) -> str:
    """Advanced agent with LLM decision, multiple tool calls, error handling."""
    try:
        llm_result, _ = real_llm_call(f"Analyze: {input_data}. Decide tools: math or search?")
        if 'math' in llm_result.lower():
            tool1_result = toolkit.math_tool("x**2 + sin(x)")
            tool2_result = toolkit.math_tool("2*x + 1")  # Multiple tools
            return f"Agent: {llm_result} | Math1: {tool1_result} | Math2: {tool2_result}"
        elif 'search' in llm_result.lower():
            return f"Agent: {llm_result} | Search: {toolkit.search_tool('AI observability')}"
        return f"Agent: {llm_result} | No tools needed."
    except Exception as e:
        raise RuntimeError(f"Agent error: {e}")

@sdk.workflow
def complex_workflow(input_data: str, reference: str = "") -> str:
    """Complex workflow with branching, multiple agents, preprocessing/postprocessing."""
    with sdk.trace_block("preprocessing", attributes={"step": "prep", "custom.type": "advanced"}):
        preprocessed = input_data.lower() + " [processed]"
        time.sleep(0.5)

    with sdk.trace_block("branching_logic", attributes={"step": "branch"}):
        if "math" in preprocessed:
            agent_result = advanced_agent(preprocessed)
        else:
            agent_result = advanced_agent(preprocessed + " fallback")

    with sdk.trace_block("secondary_agent", attributes={"step": "secondary"}):
        secondary_result = advanced_agent(agent_result)  # Chain agents

    with sdk.trace_block("postprocessing", attributes={"step": "post", "custom.flag": "complex"}):
        final_result = f"Final output: {secondary_result.upper()}"
        time.sleep(0.7)

    return final_result

# End-to-End Usage Example with trace_block_llm
@sdk.workflow
def end_to_end_workflow(user_query: str) -> str:
    """End-to-end workflow: Tool calling, reasoning, and manual tracing."""
    with sdk.trace_block_llm(prompt=user_query, model="gpt-4o-mini") as span:
        # Step 1: Tool calling LLM
        tool_result, _ = tool_calling_llm(f"Handle this query with tools if needed: {user_query}")
        span.set_attribute("llm.tool_result", sdk.anonymize_text(tool_result))
        
        # Step 2: Reasoning LLM
        reasoning_result, _ = reasoning_llm(f"Based on '{tool_result}', reason about the best response.")
        span.set_attribute("llm.reasoning_result", sdk.anonymize_text(reasoning_result))
        
        # Step 3: Final synthesis
        final_prompt = f"Synthesize: Tools: {tool_result}, Reasoning: {reasoning_result}"
        final_result, _ = real_llm_call(final_prompt)
        span.set_attribute("llm.final_response", sdk.anonymize_text(final_result))
    
    return final_result

# Agent with Reasoning Example: Tracks all traces under one parent trace
@sdk.workflow
def reasoning_agent_with_tools(user_query: str) -> str:
    """Agent that uses reasoning and tools, all under one parent trace."""
    with sdk.trace_block_llm(prompt=user_query, model="gpt-4o-mini") as parent_span:
        # Step 1: Reasoning to decide actions
        reasoning_prompt = f"Query: {user_query}. Reason step-by-step: What tools do I need? Math, search, weather, or calculator?"
        reasoning_result, _ = reasoning_llm(reasoning_prompt)
        parent_span.set_attribute("agent.reasoning", sdk.anonymize_text(reasoning_result))
        
        # Step 2: Based on reasoning, call tools
        tool_results = []
        if "math" in reasoning_result.lower():
            math_res = toolkit.math_tool("x**2 + 3*x")
            tool_results.append(f"Math: {math_res}")
        if "search" in reasoning_result.lower():
            search_res = toolkit.search_tool("AI tools")
            tool_results.append(f"Search: {search_res}")
        if "weather" in reasoning_result.lower():
            weather_res = toolkit.weather_tool("Paris")
            tool_results.append(f"Weather: {weather_res}")
        if "calculate" in reasoning_result.lower():
            calc_res = toolkit.calculator_tool("2 + 3 * 4")
            tool_results.append(f"Calculator: {calc_res}")
        
        parent_span.set_attribute("agent.tool_results", sdk.anonymize_text("; ".join(tool_results)))
        
        # Step 3: Final reasoning with tool results
        final_reasoning_prompt = f"Based on reasoning '{reasoning_result}' and tools '{'; '.join(tool_results)}', provide final answer."
        final_result, _ = reasoning_llm(final_reasoning_prompt)
        parent_span.set_attribute("agent.final_answer", sdk.anonymize_text(final_result))
    
    return final_result

# Example: Using trace_block with custom attributes dictionary
@sdk.workflow
def example_trace_block_with_attributes():
    """Demonstrates how to use trace_block and pass attributes to be logged in Arize."""
    
    # Basic trace_block with attributes dict
    custom_attrs = {
        "operation.type": "data_processing",
        "user.id": "user123",
        "request.id": "req456",
        "model.version": "v1.2",
        "custom.metric": 42.5,
        "tags": ["ai", "ml", "observability"]
    }
    
    with sdk.trace_block("data_processing_step", attributes=custom_attrs):
        # Your code here - this span will have all the custom attributes
        time.sleep(0.1)  # Simulate work
        
        # You can also add more attributes dynamically within the block
        # (Note: This would require modifying the SDK to support dynamic attribute addition)
        
    # Another example with different attributes
    workflow_attrs = {
        "workflow.name": "example_workflow",
        "step.count": 3,
        "priority": "high",
        "environment": "production"
    }
    
    with sdk.trace_block("workflow_execution", attributes=workflow_attrs):
        # Nested blocks inherit parent attributes and can have their own
        with sdk.trace_block("sub_step_1", attributes={"sub.step": 1, "action": "validate"}):
            time.sleep(0.05)
            
        with sdk.trace_block("sub_step_2", attributes={"sub.step": 2, "action": "process"}):
            time.sleep(0.05)
            
        with sdk.trace_block("sub_step_3", attributes={"sub.step": 3, "action": "finalize"}):
            time.sleep(0.05)
    
    return "Trace block example completed"

# Advanced Reasoning Agent with Complex Tracing
@sdk.workflow
def advanced_reasoning_agent_with_tracing(user_query: str) -> str:
    """Advanced agent with multi-step reasoning, tool orchestration, and detailed tracing."""
    
    with sdk.trace_block("agent_initialization", attributes={
        "agent.type": "reasoning_orchestrator",
        "query.length": len(user_query),
        "timestamp": time.time()
    }):
        # Initialize agent state
        agent_state = {
            "query": user_query,
            "reasoning_steps": [],
            "tools_used": [],
            "confidence_score": 0.0
        }
    
    # Step 1: Query Analysis and Planning
    with sdk.trace_block("query_analysis", attributes={
        "step": 1,
        "phase": "analysis",
        "analysis.type": "semantic_parsing"
    }):
        analysis_prompt = f"Analyze this query semantically: '{user_query}'. Identify: intent, entities, complexity level, required tools."
        analysis_result, _ = reasoning_llm(analysis_prompt)
        agent_state["analysis"] = analysis_result
        
        # Extract entities and intent
        entities = ["math", "search", "weather", "calculation"]  # Simplified
        detected_entities = [e for e in entities if e in analysis_result.lower()]
        agent_state["detected_entities"] = detected_entities
    
    # Step 2: Multi-step Reasoning Chain
    with sdk.trace_block("reasoning_chain", attributes={
        "step": 2,
        "phase": "reasoning",
        "chain.length": 3,
        "reasoning.type": "multi_hop"
    }):
        reasoning_steps = []
        
        # Reasoning Step 1: Decompose problem
        step1_prompt = f"Based on analysis '{analysis_result}', decompose the problem into sub-tasks."
        step1_result, _ = reasoning_llm(step1_prompt)
        reasoning_steps.append(("decompose", step1_result))
        
        # Reasoning Step 2: Plan tool usage
        step2_prompt = f"Given decomposition '{step1_result}', plan which tools to use: {detected_entities}"
        step2_result, _ = reasoning_llm(step2_prompt)
        reasoning_steps.append(("plan_tools", step2_result))
        
        # Reasoning Step 3: Validate approach
        step3_prompt = f"Validate this plan: '{step2_result}'. Is it optimal? Any missing steps?"
        step3_result, _ = reasoning_llm(step3_prompt)
        reasoning_steps.append(("validate", step3_result))
        
        agent_state["reasoning_steps"] = reasoning_steps
    
    # Step 3: Tool Orchestration with Parallel Execution
    with sdk.trace_block("tool_orchestration", attributes={
        "step": 3,
        "phase": "execution",
        "tools.count": len(detected_entities),
        "execution.mode": "parallel"
    }):
        tool_results = {}
        
        for entity in detected_entities:
            with sdk.trace_block(f"tool_{entity}", attributes={
                "tool.name": entity,
                "tool.type": "utility",
                "execution.order": detected_entities.index(entity)
            }):
                if entity == "math":
                    result = toolkit.math_tool("x**2 + sin(x)")
                    tool_results["math"] = result
                elif entity == "search":
                    result = toolkit.search_tool(user_query.split()[:3])  # First 3 words
                    tool_results["search"] = result
                elif entity == "weather":
                    result = toolkit.weather_tool("current location")
                    tool_results["weather"] = result
                elif entity == "calculation":
                    result = toolkit.calculator_tool("2 * 3.14 * 5")
                    tool_results["calculation"] = result
        
        agent_state["tool_results"] = tool_results
    
    # Step 4: Synthesis and Final Reasoning
    with sdk.trace_block("synthesis_reasoning", attributes={
        "step": 4,
        "phase": "synthesis",
        "synthesis.type": "integrative",
        "confidence.threshold": 0.8
    }):
        synthesis_prompt = f"""
        Synthesize results from reasoning chain and tools:
        Reasoning: {reasoning_steps}
        Tools: {tool_results}
        Original Query: {user_query}
        
        Provide a coherent final answer with confidence score.
        """
        final_result, _ = reasoning_llm(synthesis_prompt)
        
        # Calculate confidence (simplified)
        confidence_keywords = ["confident", "certain", "definitely", "clearly"]
        confidence_score = sum(1 for kw in confidence_keywords if kw in final_result.lower()) / len(confidence_keywords)
        agent_state["confidence_score"] = confidence_score
        
        # Add final attributes
        with sdk.trace_block("final_output", attributes={
            "output.length": len(final_result),
            "confidence.score": confidence_score,
            "agent.state": "completed"
        }):
            pass
    
    return f"Advanced Agent Result: {final_result} (Confidence: {confidence_score:.2f})"

if __name__ == "__main__":
    input_data = "Integrate math functions like x^2 and search AI tools. # Email: test@example.com"
    reference = "Expected: Integrated results and search mock."
    
    try:
        # Existing complex workflow
        result = complex_workflow(input_data, reference=reference)
        print(f"Complex Workflow Result: {result}")
        
        # New: Tool Calling LLM Example
        tool_query = "Calculate the integral of x^2 and search for 'observability tools'."
        tool_result, _ = tool_calling_llm(tool_query)
        print(f"Tool Calling LLM Result: {tool_result}")
        
        # New: Reasoning LLM Example
        reasoning_query = "If a train leaves station A at 60 mph and another at 80 mph, when do they meet?"
        reasoning_result, _ = reasoning_llm(reasoning_query)
        print(f"Reasoning LLM Result: {reasoning_result}")
        
        # New: End-to-End Workflow
        e2e_query = "Plan a trip to Paris: calculate costs, search weather, reason about itinerary."
        e2e_result = end_to_end_workflow(e2e_query)
        print(f"End-to-End Workflow Result: {e2e_result}")
        
        # New: Reasoning Agent with Tools
        agent_query = "What's the weather in Paris, calculate 100 + 50, and search for Eiffel Tower."
        agent_result = reasoning_agent_with_tools(agent_query)
        print(f"Reasoning Agent Result: {agent_result}")
        
        # New: Agent with Reasoning Example
        reasoning_agent_query = "Analyze the impact of climate change and suggest actions."
        reasoning_agent_result = reasoning_agent_with_tools(reasoning_agent_query)
        print(f"Reasoning Agent Result: {reasoning_agent_result}")
        
        # New: Trace Block Example with Attributes
        trace_result = example_trace_block_with_attributes()
        print(f"Trace Block Example Result: {trace_result}")
        
        # New: Advanced Reasoning Agent with Tracing
        advanced_tracing_query = "Plan a conference: find location, calculate budget, schedule speakers."
        advanced_tracing_result = advanced_reasoning_agent_with_tracing(advanced_tracing_query)
        print(f"Advanced Reasoning Agent Result: {advanced_tracing_result}")
        
    except Exception as e:
        print(f"Error in workflow: {e}")
    
    sdk.shutdown()
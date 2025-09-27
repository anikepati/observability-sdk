# observability_sdk.py (Enhanced Full Version; Direct Annotations via log_evaluations, Fallback to Attributes in Phoenix Mode; Updated for OpenShift HTTPS, No Cert Validation, No Proxy)
import logging
import random
import pandas as pd
from typing import Callable, Any, Dict
import json
import requests
import certifi
import httpx
from dotenv import load_dotenv
import os
import time
import threading
load_dotenv()
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter as GrpcExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter as BaseHttpExporter
from grpc import ssl_channel_credentials
from contextlib import contextmanager
# Presidio imports
from presidio_analyzer import AnalyzerEngine, PatternRecognizer
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig
# Custom HttpExporter
class CustomHttpExporter(BaseHttpExporter):
    def __init__(self, *args, verify=True, proxies=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._session = requests.Session()
        self._session.verify = verify
        self._session.proxies = proxies or {}
        logger.debug(f"CustomHttpExporter initialized with verify={verify}, proxies={self._session.proxies}")
try:
    from phoenix.otel import register as phoenix_register
    import phoenix as px
    from phoenix.trace import SpanEvaluations
except ImportError:
    phoenix_register = None
    px = None
    SpanEvaluations = None
try:
    
    from arize.pandas.logger import Client as ArizeClient
    from arize.pandas.logger import Schema
    from arize.utils.types import ModelTypes, Environments
except ImportError:
    
    ArizeClient = None
    Schema = None
    ModelTypes = None
    Environments = None
from phoenix.evals import (
    HallucinationEvaluator,
    QAEvaluator,
    RelevanceEvaluator,
    ToxicityEvaluator,
    OpenAIModel,
    run_evals,
)
from rouge_score import rouge_scorer
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
custom_cert = os.getenv('CUSTOM_SSL_CERT_FILE', certifi.where())
os.environ['REQUESTS_CA_BUNDLE'] = custom_cert
logger.debug(f"SSL cert set: {custom_cert}")
class PIIEvaluator:
    def evaluate(self, query: str = None, response: str = None, reference: str = None, sleep_time_in_seconds: int = 0):
        analyzer = AnalyzerEngine()
        texts = {'input': query or '', 'output': response or '', 'reference': reference or ''}
        detected = {}
        for location, text in texts.items():
            if text:
                results = analyzer.analyze(text=text, language='en')
                if results:
                    detected[location] = [(res.entity_type, text[res.start:res.end]) for res in results]
        has_pii = bool(detected)
        label = "has_pii" if has_pii else "no_pii"
        score = 1 if has_pii else 0
        explanation = f"PII detected: {detected}" if has_pii else "No PII detected"
        return {"label": label, "score": score, "explanation": explanation}
class FairnessEvaluator:
    def evaluate(self, query: str = None, response: str = None, reference: str = None, sleep_time_in_seconds: int = 0):
        # Simple stub for fairness: Check for biased terms (extend with real bias detection libraries like Fairlearn)
        biased_terms = ['bias', 'discriminate', 'unfair']  # Placeholder; use advanced models in production
        text = f"{query or ''} {response or ''} {reference or ''}"
        has_bias = any(term in text.lower() for term in biased_terms)
        label = "biased" if has_bias else "fair"
        score = 1 if has_bias else 0
        explanation = f"Biased terms detected: {has_bias}"
        return {"label": label, "score": score, "explanation": explanation}
class CustomOpenAIModel(OpenAIModel):
    def __init__(self, *args, custom_headers: Dict[str, str] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.custom_headers = custom_headers or {}
    def _generate(self, prompt: str, **kwargs: Dict[str, Any]) -> Dict[str, Any]:
        logger.info(f"Sending chat completion with custom headers: {self.custom_headers}")
        return super()._generate(prompt, **kwargs)
class ObservabilitySDK:
    _instance = None
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ObservabilitySDK, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
    def _initialize(self):
        # ... existing code ...
        self.apigee_manager = ApigeeManager(
            apigee_endpoint=os.getenv('APIGEE_ENDPOINT', 'https://your-apigee-endpoint/oauth/token'),
            client_id=os.getenv('APIGEE_CLIENT_ID', 'your_client_id'),
            client_secret=os.getenv('APIGEE_CLIENT_SECRET', 'your_client_secret'),
            refresh_interval=int(os.getenv('APIGEE_REFRESH_INTERVAL', 3600))
        )
        # ... rest of _initialize ...
        self.stored_traces = []
        self.eval_model = self.setup_eval_model()
        self.evaluators = self.setup_evaluators()
        self.anonymizer = AnonymizerEngine()
        # Load spaCy model from local path to avoid global installation
        import spacy
        model_path = os.path.join(os.path.dirname(__file__), 'models', 'en_core_web_sm')
        if os.path.exists(model_path):
            self.nlp = spacy.load(model_path)
            self.analyzer = AnalyzerEngine(nlp_engine=self.nlp)
            logger.info("Loaded en_core_web_sm from local path.")
        else:
            logger.warning("Local en_core_web_sm model not found at {}, falling back to installed version.".format(model_path))
            self.analyzer = AnalyzerEngine()
        self.analyzer.registry.add_recognizer(PatternRecognizer(supported_entity="CUSTOM_EMAIL", patterns=[r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b']))
        self.model_name = "gpt-4o"  # Updated to 2025 model
        # Start runtime refresh timer
        self._refresh_timer = None
        self._start_refresh_timer()
    def _start_refresh_timer(self):
        if self.refresh_interval > 0:
            self.refresh_config()
            self._refresh_timer = threading.Timer(self.refresh_interval, self._start_refresh_timer)
            self._refresh_timer.daemon = True
            self._refresh_timer.start()
    def refresh_config(self):
        """Refresh Apigee token and custom headers at runtime."""
        try:
            new_headers = self.apigee_manager.get_headers()
            if new_headers != self.custom_headers:
                logger.info("Refreshing custom headers from Apigee.")
                self.custom_headers = new_headers
                self.eval_model = self.setup_eval_model()
                self.evaluators = self.setup_evaluators()
        except Exception as e:
            logger.warning(f"Failed to refresh Apigee config: {e}")
    def setup_eval_model(self):
        try:
            from openai import OpenAI
        except ImportError:
            logger.error("openai package not installed; evaluations will fail without OPENAI_API_KEY")
            raise ImportError("openai package required for evaluations")
        api_key = os.getenv('OPENAI_API_KEY')  # Use static key, headers from manager
        base_url = self.custom_llm_url or 'https://api.openai.com/v1'
        headers = self.apigee_manager.get_headers()  # Get fresh headers with token
        logger.info(f"Using custom LLM gateway: base_url={base_url}, with Apigee headers: {list(headers.keys())}")
        openai_client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            default_headers=headers,
            http_client=httpx.Client(proxies=self.proxies, verify=not self.insecure)
        )
        return CustomOpenAIModel(model=self.model_name, openai_client=openai_client, custom_headers=headers)
    def setup_tracer(self):
        """
        Sets up the OpenTelemetry tracer provider and exporter.
        Supports:
        - gRPC exporter if endpoint starts with 'grpc://' or is not HTTP(S)
        - HTTP/HTTPS exporter otherwise
        - SSL validation can be disabled via self.insecure
        """
        tracer_provider = TracerProvider()
        endpoint = self.otlp_endpoint if self.mode == 'local' else self.ax_endpoint
        # Determine exporter type
        if endpoint.startswith('http://') or endpoint.startswith('https://'):
            # Use HTTP/HTTPS exporter, ignore SSL validation if insecure
            exporter = CustomHttpExporter(
                endpoint=endpoint,
                verify=not self.insecure,
                proxies=self.proxies
            )
            logger.debug(f"Using HTTP/HTTPS exporter for endpoint: {endpoint}, SSL verify: {not self.insecure}")
        else:
            # Use gRPC exporter, ignore SSL validation if insecure
            credentials = None if self.insecure else ssl_channel_credentials()
            exporter = GrpcExporter(
                endpoint=endpoint,
                credentials=credentials
            )
            logger.debug(f"Using gRPC exporter for endpoint: {endpoint}, SSL verify: {not self.insecure}")
        processor = BatchSpanProcessor(exporter)
        tracer_provider.add_span_processor(processor)
        return tracer_provider
    def setup_client(self):
        if self.mode == 'local':
            if px:
                self.phoenix_client = px.launch_app()
                logger.info("Phoenix client launched for local mode.")
        else:
            if ArizeClient:
                self.arize_client = ArizeClient(
                    space_key=os.getenv('ARIZE_SPACE_KEY'),
                    api_key=os.getenv('ARIZE_API_KEY')
                )
                logger.info("Arize client initialized for AX mode.")
        self.stored_traces = []
        self.eval_model = self.setup_eval_model()
        self.evaluators = self.setup_evaluators()
        self.anonymizer = AnonymizerEngine()
        # Load spaCy model from local path to avoid global installation
        import spacy
        model_path = os.path.join(os.path.dirname(__file__), 'models', 'en_core_web_sm')
        if os.path.exists(model_path):
            self.nlp = spacy.load(model_path)
            self.analyzer = AnalyzerEngine(nlp_engine=self.nlp)
            logger.info("Loaded en_core_web_sm from local path.")
        else:
            logger.warning("Local en_core_web_sm model not found at {}, falling back to installed version.".format(model_path))
            self.analyzer = AnalyzerEngine()
        self.analyzer.registry.add_recognizer(PatternRecognizer(supported_entity="CUSTOM_EMAIL", patterns=[r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b']))
        self.model_name = "gpt-4o"  # Updated to 2025 model
        # Start runtime refresh timer
        self._refresh_timer = None
        self._start_refresh_timer()
    def setup_evaluators(self):
        return [
            QAEvaluator(self.eval_model),
            HallucinationEvaluator(self.eval_model),
            RelevanceEvaluator(self.eval_model),
            ToxicityEvaluator(self.eval_model),
            PIIEvaluator(),
            FairnessEvaluator(),
        ]
    def _retry_operation(self, operation: Callable, max_retries: int = 3) -> bool:
        for attempt in range(max_retries):
            try:
                operation()
                return True
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    logger.error(f"Operation failed after {max_retries} attempts.")
                    return False
    def _add_score_if_missing(self, eval_df: pd.DataFrame, eval_name: str) -> pd.DataFrame:
        if 'score' not in eval_df.columns:
            eval_df['score'] = 0.0
        return eval_df
    def log_evaluation(self, eval_df: pd.DataFrame, eval_name: str, offline: bool = False) -> bool:
        eval_df = self._add_score_if_missing(eval_df, eval_name)
        if 'explanation' in eval_df.columns:
            eval_df['explanation'] = eval_df['explanation'].apply(self.anonymize_text)
        eval_df = eval_df.rename_axis("context.span_id")
        def log_op():
            if self.mode == 'local':
                if self.phoenix_client:
                    self.phoenix_client.log_evaluations(SpanEvaluations(eval_name=eval_name, dataframe=eval_df))
            else:
                if self.arize_client:
                    schema = Schema(
                        prediction_id_column_name="context.span_id",
                        prediction_score_column_name="score",
                        prediction_label_column_name="label",
                        actual_label_column_name="label",
                    )
                    self.arize_client.log(
                        dataframe=eval_df,
                        model_id="gen-ai-eval-model",
                        model_version="1.0",
                        model_type=ModelTypes.SCORE_CATEGORICAL,
                        environment=Environments.PRODUCTION if not offline else Environments.VALIDATION,
                        schema=schema
                    )
        return self._retry_operation(log_op)
    def anonymize_text(self, text: str) -> str:
        if not text:
            return text
        results = self.analyzer.analyze(text=text, language='en')
        anonymized = self.anonymizer.anonymize(
            text=text,
            analyzer_results=results,
            operators={"DEFAULT": OperatorConfig("replace", {"new_value": "[REDACTED]"})}
        ).text
        return anonymized
    def log_rouge_evaluation(self, rouge_df: pd.DataFrame, offline: bool = False) -> bool:
        def log_op():
            if self.mode == 'local':
                if self.phoenix_client:
                    self.phoenix_client.log_evaluations(SpanEvaluations(eval_name="ROUGE", dataframe=rouge_df))
            else:
                if self.arize_client:
                    schema = Schema(
                        prediction_id_column_name="context.span_id",
                        prediction_score_column_name="score",
                    )
                    self.arize_client.log(
                        dataframe=rouge_df,
                        model_id="gen-ai-rouge-model",
                        model_version="1.0",
                        model_type=ModelTypes.NUMERIC,
                        environment=Environments.PRODUCTION if not offline else Environments.VALIDATION,
                        schema=schema
                    )
        return self._retry_operation(log_op)
    @contextmanager
    def trace_block(self, name: str, attributes: dict = None):
        start_time = time.time()
        span = self.tracer.start_as_current_span(name, attributes=attributes or {})
        try:
            yield span
        except Exception as e:
            span.record_exception(e)
            span.set_attribute("error.type", type(e).__name__)
            span.set_attribute("error.message", self.anonymize_text(str(e)))
            raise
        finally:
            duration = time.time() - start_time
            span.set_attribute("duration_ms", duration * 1000)
            span.set_attribute("component", name.lower())
            span.end()
    @contextmanager
    def trace_block_llm(self, prompt: str, model: str = None, **kwargs):
        """
        Context manager for tracing LLM workflows, tracking before and after execution.
        Sets LLM-specific attributes and events.
        """
        span = self.tracer.start_as_current_span("llm_workflow")
        span.set_attribute("llm.prompt", self.anonymize_text(prompt))
        span.set_attribute("llm.model", model or "unknown")
        for key, value in kwargs.items():
            span.set_attribute(f"llm.custom.{key}", value)
        span.add_event("LLM Execution Started", attributes={"timestamp": time.time()})
        try:
            yield span
        except Exception as e:
            span.record_exception(e)
            span.set_attribute("error.type", type(e).__name__)
            span.set_attribute("error.message", self.anonymize_text(str(e)))
            raise
        finally:
            span.add_event("LLM Execution Completed", attributes={"timestamp": time.time()})
            span.end()
    def workflow(self, func: Callable) -> Callable:
        def wrapper(*args, **kwargs) -> Any:
            start_time = time.time()
            with self.tracer.start_as_current_span(f"workflow_{func.__name__}") as parent_span:
                try:
                    input_text = args[0] if args else kwargs.get('input', '')
                    reference = kwargs.get('reference', '')
                    parent_span.set_attribute("input.value", self.anonymize_text(input_text))
                    parent_span.set_attribute("reference.value", self.anonymize_text(reference))

                    result = func(*args, **kwargs)

                    duration = time.time() - start_time
                    parent_span.set_attribute("duration_ms", duration * 1000)
                    parent_span.set_attribute("output.value", self.anonymize_text(result))

                    self.run_online_evals(parent_span.context.span_id, input_text, reference, result, parent_span)
                    self.stored_traces.append({
                        'span_id': parent_span.context.span_id,
                        'input': input_text,
                        'output': result,
                        'reference': reference,
                        'duration': duration,
                        'tokens': len(input_text.split()) + len(result.split())  # Approximate
                    })
                    return result
                except Exception as e:
                    parent_span.record_exception(e)
                    parent_span.set_attribute("error.type", type(e).__name__)
                    parent_span.set_attribute("error.message", self.anonymize_text(str(e)))
                    raise
        return wrapper
    def tool_span(self, func: Callable) -> Callable:
        def wrapper(*args, **kwargs) -> Any:
            start_time = time.time()
            with self.tracer.start_as_current_span(f"tool_{func.__name__}") as span:
                try:
                    result = func(*args, **kwargs)
                    input_data = args[0] if args else kwargs.get('input_data', '')
                    duration = time.time() - start_time
                    span.set_attribute("duration_ms", duration * 1000)
                    span.set_attribute("input.value", self.anonymize_text(input_data))
                    span.set_attribute("output.value", self.anonymize_text(result))
                    return result
                except Exception as e:
                    span.record_exception(e)
                    span.set_attribute("error.type", type(e).__name__)
                    span.set_attribute("error.message", self.anonymize_text(str(e)))
                    raise
        return wrapper
    def llm_span(self, func: Callable) -> Callable:
        def wrapper(*args, **kwargs) -> Any:
            start_time = time.time()
            with self.tracer.start_as_current_span(f"llm_{func.__name__}") as span:
                try:
                    response = func(*args, **kwargs)
                    if isinstance(response, tuple) and len(response) == 2:
                        result, token_info = response
                        model = token_info.get('model', 'unknown')
                        input_tokens = token_info.get('input_tokens', 0)
                        output_tokens = token_info.get('output_tokens', 0)
                        custom_attrs = token_info.get('custom_attrs', {})
                    else:
                        result = response
                        model = 'unknown'
                        input_tokens = 0
                        output_tokens = 0
                        custom_attrs = {}

                    duration = time.time() - start_time
                    input_prompt = args[0] if args else kwargs.get('prompt', '')

                    span.set_attribute("duration_ms", duration * 1000)
                    span.set_attribute("llm.model", model)
                    span.set_attribute("llm.input_tokens", input_tokens)
                    span.set_attribute("llm.output_tokens", output_tokens)
                    span.set_attribute("llm.total_tokens", input_tokens + output_tokens)
                    span.set_attribute("input.value", self.anonymize_text(input_prompt))
                    span.set_attribute("output.value", self.anonymize_text(result))

                    for key, value in custom_attrs.items():
                        span.set_attribute(f"llm.custom.{key}", value)

                    return result
                except Exception as e:
                    span.record_exception(e)
                    span.set_attribute("error.type", type(e).__name__)
                    span.set_attribute("error.message", self.anonymize_text(str(e)))
                    raise
        return wrapper
    def agent_span(self, func: Callable) -> Callable:
        def wrapper(*args, **kwargs) -> Any:
            start_time = time.time()
            with self.tracer.start_as_current_span(f"agent_{func.__name__}") as span:
                try:
                    result = func(*args, **kwargs)
                    input_data = args[0] if args else kwargs.get('input_data', '')
                    duration = time.time() - start_time
                    span.set_attribute("duration_ms", duration * 1000)
                    span.set_attribute("input.value", self.anonymize_text(input_data))
                    span.set_attribute("output.value", self.anonymize_text(result))
                    return result
                except Exception as e:
                    span.record_exception(e)
                    span.set_attribute("error.type", type(e).__name__)
                    span.set_attribute("error.message", self.anonymize_text(str(e)))
                    raise
        return wrapper
    def run_online_evals(self, span_id, input_text, reference, output, span=None):
        dataframe_original = pd.DataFrame([{"input": input_text, "output": output, "reference": reference}])
        pii_eval = next((e for e in self.evaluators if isinstance(e, PIIEvaluator)), None)
        fairness_eval = next((e for e in self.evaluators if isinstance(e, FairnessEvaluator)), None)
        other_evaluators = [e for e in self.evaluators if not isinstance(e, (PIIEvaluator, FairnessEvaluator))]
        eval_results = []
        eval_names = []
        sampled_evals = []
        if pii_eval:
            pii_results = run_evals(dataframe=dataframe_original, evaluators=[pii_eval], provide_explanation=True)[0]
            pii_results['sampled'] = False
            eval_results.append(pii_results)
            eval_names.append("PII")
            sampled_evals.append(False)
            if self.sample_tracking_enabled:
                span.set_attribute("evaluation.type.PII", "full")
                span.set_attribute("evaluation.sample_rate", self.sample_rate)
                span.set_attribute("evaluation.sample_value.PII", 1.0)  # Always evaluated
                span.set_attribute("evaluation.sampled.PII", False)
                span.add_event("Eval Triggered", attributes={"eval_name": "PII", "sampled": False, "sample_value": 1.0})
        if fairness_eval:
            fairness_results = [fairness_eval.evaluate(query=input_text, response=output, reference=reference)]
            fairness_df = pd.DataFrame(fairness_results)
            fairness_df['sampled'] = False
            eval_results.append(fairness_df)
            eval_names.append("Fairness")
            sampled_evals.append(False)
            if self.sample_tracking_enabled:
                span.set_attribute("evaluation.type.Fairness", "full")
                span.set_attribute("evaluation.sample_value.Fairness", 1.0)  # Always evaluated
                span.set_attribute("evaluation.sampled.Fairness", False)
                span.add_event("Eval Triggered", attributes={"eval_name": "Fairness", "sampled": False, "sample_value": 1.0})
        input_anon = self.anonymize_text(input_text)
        output_anon = self.anonymize_text(output)
        reference_anon = self.anonymize_text(reference)
        dataframe_anon = pd.DataFrame([{"input": input_anon, "output": output_anon, "reference": reference_anon}])
        for evaluator in other_evaluators:
            eval_name = type(evaluator).__name__.replace('Evaluator', '')
            is_sampled = isinstance(evaluator, (QAEvaluator, HallucinationEvaluator, RelevanceEvaluator, ToxicityEvaluator))
            sample_value = random.random()
            sampled = sample_value <= self.sample_rate if is_sampled else False
            if is_sampled and not sampled:
                continue
            results = run_evals(dataframe=dataframe_anon, evaluators=[evaluator], provide_explanation=True)[0]
            results['sampled'] = sampled
            eval_results.append(results)
            eval_names.append(eval_name)
            sampled_evals.append(sampled)
            if self.sample_tracking_enabled:
                eval_type = "sampled" if sampled else "full"
                span.set_attribute(f"evaluation.type.{eval_name}", eval_type)
                span.set_attribute("evaluation.sample_rate", self.sample_rate)
                span.set_attribute(f"evaluation.sample_value.{eval_name}", sample_value)
                span.set_attribute(f"evaluation.sampled.{eval_name}", sampled)
                span.add_event("Eval Triggered", attributes={"eval_name": eval_name, "sampled": sampled, "sample_value": sample_value, "reason": "Random sample met" if sampled else "Full eval"})
        for eval_df, eval_name in zip(eval_results, eval_names):
            eval_df.index = pd.Index([span_id], name="context.span_id")
            # Direct annotations via log_evaluations (primary for Phoenix)
            self.log_evaluation(eval_df, eval_name)
            # Fallback: Add to attributes (visible in Phoenix as span metadata)
            if 'score' in eval_df.columns:
                span.set_attribute(f"eval.{eval_name}.score", eval_df['score'].iloc[0])
            if 'label' in eval_df.columns:
                span.set_attribute(f"eval.{eval_name}.label", eval_df['label'].iloc[0])
            if 'explanation' in eval_df.columns:
                span.set_attribute(f"eval.{eval_name}.explanation", self.anonymize_text(eval_df['explanation'].iloc[0]))
        if reference:
            rouge_df = self.compute_rouge(span_id, reference, output)
            # Direct for Phoenix
            self.log_rouge_evaluation(rouge_df)
            # Fallback attributes
            span.set_attribute("eval.ROUGE.score", rouge_df['score'].iloc[0])
            span.set_attribute("eval.ROUGE.explanation", rouge_df['explanation'].iloc[0])
            if self.sample_tracking_enabled:
                span.set_attribute("evaluation.type.ROUGE", "full")
                span.set_attribute("evaluation.sample_rate", self.sample_rate)
                span.set_attribute("evaluation.sample_value.ROUGE", 1.0)  # Always evaluated
                span.set_attribute("evaluation.sampled.ROUGE", False)
                span.add_event("Eval Triggered", attributes={"eval_name": "ROUGE", "sampled": False, "sample_value": 1.0})
    def run_offline_evals(self):
        if not self.stored_traces:
            return
        inputs = [trace['input'] for trace in self.stored_traces]
        outputs = [trace['output'] for trace in self.stored_traces]
        references = [trace['reference'] for trace in self.stored_traces]
        span_ids = [trace['span_id'] for trace in self.stored_traces]
        durations = [trace['duration'] for trace in self.stored_traces]
        tokens = [trace['tokens'] for trace in self.stored_traces]
        dataframe_original = pd.DataFrame({"input": inputs, "output": outputs, "reference": references})
        pii_eval = next((e for e in self.evaluators if isinstance(e, PIIEvaluator)), None)
        fairness_eval = next((e for e in self.evaluators if isinstance(e, FairnessEvaluator)), None)
        other_evaluators = [e for e in self.evaluators if not isinstance(e, (PIIEvaluator, FairnessEvaluator))]
        eval_results = []
        eval_names = []
        if pii_eval:
            pii_results = run_evals(dataframe=dataframe_original, evaluators=[pii_eval], provide_explanation=True)[0]
            eval_results.append(pii_results)
            eval_names.append("PII")
        if fairness_eval:
            fairness_results = []
            for i in range(len(inputs)):
                res = fairness_eval.evaluate(query=inputs[i], response=outputs[i], reference=references[i])
                fairness_results.append(res)
            fairness_df = pd.DataFrame(fairness_results)
            eval_results.append(fairness_df)
            eval_names.append("Fairness")
        inputs_anon = [self.anonymize_text(i) for i in inputs]
        outputs_anon = [self.anonymize_text(o) for o in outputs]
        references_anon = [self.anonymize_text(r) for r in references]
        dataframe_anon = pd.DataFrame({"input": inputs_anon, "output": outputs_anon, "reference": references_anon})
        for evaluator in other_evaluators:
            results = run_evals(dataframe=dataframe_anon, evaluators=[evaluator], provide_explanation=True)[0]
            eval_results.append(results)
            eval_names.append(type(evaluator).__name__.replace('Evaluator', ''))
        for eval_df, eval_name in zip(eval_results, eval_names):
            eval_df.index = pd.Index(span_ids, name="context.span_id")
            self.log_evaluation(eval_df, eval_name, offline=True)
        has_ref_mask = [bool(r) for r in references]
        if any(has_ref_mask):
            rouge_df = self.compute_rouge_batch([s for s, m in zip(span_ids, has_ref_mask) if m],
                                                [r for r, m in zip(references, has_ref_mask) if m],
                                                [o for o, m in zip(outputs, has_ref_mask) if m])
            self.log_rouge_evaluation(rouge_df, offline=True)
        logger.info(f"Offline stats: Avg duration {sum(durations)/len(durations):.2f}s, Avg tokens {sum(tokens)/len(tokens):.0f}")
        self.stored_traces = []
    def compute_rouge(self, span_id, reference, output):
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        scores = scorer.score(reference, output)
        score = (scores['rouge1'].fmeasure + scores['rouge2'].fmeasure + scores['rougeL'].fmeasure) / 3
        df = pd.DataFrame({"score": [score], "explanation": [str(scores)]})
        df.index = pd.Index([span_id], name="context.span_id")
        return df
    def compute_rouge_batch(self, span_ids, references, outputs):
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        scores_list = []
        explanations = []
        for ref, out in zip(references, outputs):
            scores = scorer.score(ref, out)
            score = (scores['rouge1'].fmeasure + scores['rouge2'].fmeasure + scores['rougeL'].fmeasure) / 3
            scores_list.append(score)
            explanations.append(str(scores))
        df = pd.DataFrame({"score": scores_list, "explanation": explanations})
        df.index = pd.Index(span_ids, name="context.span_id")
        return df
    def shutdown(self):
        if self._refresh_timer:
            self._refresh_timer.cancel()
        self.run_offline_evals()
        self.tracer_provider.shutdown()
        if self.mode == 'local' and self.phoenix_client:
            px.close_app()
class ApigeeManager:
    """Separate class to manage Apigee API key fetching and caching."""
    
    def __init__(self, apigee_endpoint: str, client_id: str, client_secret: str, refresh_interval: int = 3600):
        self.apigee_endpoint = apigee_endpoint
        self.client_id = client_id
        self.client_secret = client_secret
        self.refresh_interval = refresh_interval
        self._last_refresh = 0
        self._token = None
        self._headers = {}
    
    def get_token(self) -> str:
        """Fetch or return cached Apigee token."""
        current_time = time.time()
        if self._token is None or (current_time - self._last_refresh) > self.refresh_interval:
            self._refresh_token()
        return self._token
    
    def get_headers(self) -> Dict[str, str]:
        """Get headers including the token."""
        token = self.get_token()
        return {"Authorization": f"Bearer {token}", **self._headers}
    
    def _refresh_token(self):
        """Call Apigee API to get new token."""
        try:
            response = requests.post(
                self.apigee_endpoint,
                data={
                    "grant_type": "client_credentials",
                    "client_id": self.client_id,
                    "client_secret": self.client_secret
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"}
            )
            response.raise_for_status()
            data = response.json()
            self._token = data.get("access_token")
            self._last_refresh = time.time()
            logger.info("Apigee token refreshed successfully.")
        except Exception as e:
            logger.error(f"Failed to refresh Apigee token: {e}")
            raise
"""Specifications of all agents, including those planned for Assignment 4."""

from kazner_agents.agents.base import AgentSpec

ORCHESTRATOR = AgentSpec(
    name="Orchestrator",
    role="Plan batches, route messages, enforce limits",
    inputs="Task (CollectRequest)",
    outputs="RunSummary",
    tools=("state store",),
    done_when="All batches have an EvaluationReport, or a limit is hit",
    stage="A2",
)

COLLECTOR = AgentSpec(
    name="Collector",
    role="Fetch text, split into sentences and words, record provenance",
    inputs="CollectRequest",
    outputs="DocumentBatch (one per batch)",
    tools=("Wikipedia API", "file reader", "sentence/word splitter"),
    done_when="All requested sentences emitted with source URL and licence",
    stage="A2",
)

PRE_ANNOTATOR = AgentSpec(
    name="PreAnnotator",
    role="Label words with the fine-tuned mBERT",
    inputs="DocumentBatch",
    outputs="Annotation (annotator = mbert, IOB2 labels)",
    tools=("mBERT inference",),
    done_when="Every word has a label",
    stage="A4",
)

LLM_ANNOTATOR = AgentSpec(
    name="LLMAnnotator",
    role="Label words with an LLM using few-shot KazNERD examples",
    inputs="DocumentBatch",
    outputs="Annotation (annotator = llm, spans)",
    tools=("LLM", "few-shot example store"),
    done_when="Valid JSON for every sentence",
    stage="A2",
)

BOUNDARY_NORMALIZER = AgentSpec(
    name="BoundaryNormalizer",
    role="Convert spans to word-level IOB2, repair invalid spans and sequences, validate labels",
    inputs="Annotation (spans or IOB2 labels)",
    outputs="NormalizedAnnotation",
    tools=("IOB2 validator", "label set"),
    done_when="Output passes validation",
    stage="A2",
)

VERIFIER = AgentSpec(
    name="Verifier",
    role="Compare the two annotations; accept agreement, adjudicate or escalate disagreement",
    inputs="2 x NormalizedAnnotation",
    outputs="VerificationResult",
    tools=("LLM (adjudication)", "guideline rules", "human-review queue"),
    done_when="Every disagreement is resolved or escalated",
    stage="A4",
)

EVALUATOR = AgentSpec(
    name="Evaluator",
    role="Score against gold when available; otherwise report agreement or a summary; export",
    inputs="VerificationResult (+ gold); in A2 NormalizedAnnotation",
    outputs="EvaluationReport, exported files",
    tools=("state store", "file exporter", "kazner evaluator (A4)"),
    done_when="Report and exports written",
    stage="A2",
)

ALL_SPECS = (
    ORCHESTRATOR,
    COLLECTOR,
    PRE_ANNOTATOR,
    LLM_ANNOTATOR,
    BOUNDARY_NORMALIZER,
    VERIFIER,
    EVALUATOR,
)

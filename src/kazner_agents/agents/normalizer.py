"""BoundaryNormalizer: annotation (spans or labels) -> valid word-level IOB2."""

from __future__ import annotations

from kazner_agents.agents.base import Agent
from kazner_agents.agents.specs import BOUNDARY_NORMALIZER
from kazner_agents.context import RunContext
from kazner_agents.errors import AgentError
from kazner_agents.iob2 import repair_iob2, spans_to_iob2, validate_iob2
from kazner_agents.messages import Annotation, NormalizedAnnotation, SentenceLabels


def validate_sentences(label_lists: list[list[str]]) -> list[str]:
    """Tool: the IOB2 validator, run once over all sentences of a batch."""
    problems = []
    for index, labels in enumerate(label_lists):
        problems += [f"sentence {index}: {problem}" for problem in validate_iob2(labels)]
    return problems


class BoundaryNormalizer(Agent):
    spec = BOUNDARY_NORMALIZER

    def handle(self, payload: Annotation, ctx: RunContext) -> list[NormalizedAnnotation]:
        sentences = []
        repairs_applied = 0
        for sentence in payload.sentences:
            if sentence.entities is not None:
                labels, repairs = spans_to_iob2(sentence.words, sentence.entities)
            else:
                labels, repairs = repair_iob2(sentence.labels)
            repairs_applied += len(repairs)
            # Notes from the annotator also send the sentence to human review.
            notes = [f"annotator: {note}" for note in sentence.notes]
            sentences.append(
                SentenceLabels(
                    sentence_id=sentence.sentence_id,
                    words=sentence.words,
                    labels=labels,
                    repairs=notes + repairs,
                )
            )

        problems = ctx.call_tool(
            self.name, "iob2_validator", validate_sentences, [s.labels for s in sentences]
        )
        if problems:
            raise AgentError("normalised labels are still invalid: " + "; ".join(problems[:5]))

        return [
            NormalizedAnnotation(
                batch_id=payload.batch_id,
                annotator=payload.annotator,
                sentences=sentences,
                repairs_applied=repairs_applied,
            )
        ]

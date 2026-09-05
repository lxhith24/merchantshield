"""The single LLM agent: a bounded, tool-selecting, evidence-grounded investigator.

The loop below is the whole agent. It forms a ring hypothesis and at least one
legitimate alternative, picks the next read-only tool from what it still does
not know, and stops when the evidence is sufficient or a budget runs out.

It may recommend or abstain. It may never compute a risk score, approve a
merchant, or reject one.
"""
from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Dict, List, Mapping, Optional, Protocol, Sequence, Tuple

from pydantic import ValidationError

from ..agent.contracts import CandidateAssessment
from ..analysis.evidence_graph import RingCandidate
from ..models.merchant import MerchantApplication
from .contracts import (
    GroundingStatus,
    Hypothesis,
    HypothesisKind,
    InformationRequestItem,
    InvestigationBudget,
    InvestigationResult,
    InvestigationStatus,
    InvestigatorOutput,
    Recommendation,
    ToolCallStatus,
    ToolObservation,
)
from .grounding import CitationValidator
from .memory import CaseMemory
from .tools import (
    TOOL_DESCRIPTIONS,
    TOOL_NAMES,
    InvestigationToolbox,
    ToolError,
    build_evidence_catalogue,
    disclosed_evidence_ids,
)

SYSTEM_PROMPT = """\
You are MerchantShield's onboarding-ring investigator. You explain evidence that
deterministic experts have already produced. You never compute or revise a risk
score, and you never approve or reject a merchant.

Your question is narrow: are the observed links between these merchant
applications better explained by a COORDINATED RING, or by a LEGITIMATE shared
arrangement such as an accountant filing for clients, a franchise, a coworking
office, family ownership, or a shared kiosk?

Rules:
- Form a ring hypothesis AND at least one specific legitimate alternative.
- Choose the next tool from what you still do not know. Do not call every tool.
- Cite only evidence IDs that a tool call actually returned to you. Inventing an
  evidence ID invalidates your entire narrative.
- If a decisive fact is missing, recommend request_information and say what you
  need. If you cannot reach a defensible view, abstain.

Respond with a single JSON object and nothing else. To use a tool:
  {"action": "call_tool", "tool": "<name>", "arguments": {...}}
To finish:
  {"action": "conclude", "recommendation": "continue_onboarding|request_information|human_review|abstain",
   "ring_hypothesis": {"label": "...", "statement": "...",
                       "supporting_evidence_ids": [...], "contradicting_evidence_ids": [...]},
   "legitimate_alternatives": [{"label": "...", "statement": "...",
                                "supporting_evidence_ids": [...], "contradicting_evidence_ids": [...]}],
   "missing_evidence": ["..."],
   "requested_information": [{"item_code": "...", "description": "...", "member_id": null}],
   "confidence": 0.0,
   "cited_evidence_ids": ["..."],
   "narrative": "..."}

`confidence` describes your recommendation, not the merchant's risk.
"""


class ProviderDisabled(Exception):
    """No LLM is configured. The case abstains and goes to a human."""


class ProviderError(Exception):
    """The provider could not produce a response."""


class InvestigatorProvider(Protocol):
    name: str
    version: str

    async def propose(
        self,
        *,
        system: str,
        transcript: Sequence[Mapping[str, Any]],
        max_tokens: int,
    ) -> str: ...


class DisabledProvider:
    """Explicit LLM-disabled mode; produces a labelled abstention, not a score."""

    name = "llm_disabled"
    version = "disabled-v1"

    async def propose(self, *, system, transcript, max_tokens) -> str:
        raise ProviderDisabled("No investigator model is configured.")


class ScriptedProvider:
    """Replays fixed responses so failure modes can be tested exactly."""

    name = "scripted"
    version = "scripted-v1"

    def __init__(self, responses: Sequence[str]) -> None:
        self._responses = list(responses)
        self.calls: List[Sequence[Mapping[str, Any]]] = []

    async def propose(self, *, system, transcript, max_tokens) -> str:
        self.calls.append(list(transcript))
        if not self._responses:
            raise ProviderError("scripted provider exhausted")
        return self._responses.pop(0)


class DeterministicInvestigator:
    """A real investigator policy with no model behind it.

    It exists so the demonstration, the tests and the CI run are reproducible
    without an API key. It reads tool results structurally and picks the next
    tool from what is still unresolved, which is the same discipline the LLM
    prompt asks for.
    """

    name = "deterministic"
    version = "deterministic-investigator-v1"

    STRONG_ATTRIBUTES = ("bank_account", "owner_pan", "device_fingerprint")

    async def propose(self, *, system, transcript, max_tokens) -> str:
        brief = _first_of_type(transcript, "case_brief") or {}
        results = _tool_results(transcript)
        called = tuple(results)

        if "get_candidate_summary" not in called:
            return _action(
                "get_candidate_summary", {"candidate_id": brief.get("candidate_id")}
            )
        if "get_relationship_evidence" not in called:
            return _action(
                "get_relationship_evidence",
                {"candidate_id": brief.get("candidate_id"), "min_severity": 0.0},
            )

        evidence = tuple(results["get_relationship_evidence"].get("evidence", ()))
        strong = tuple(
            item for item in evidence if item.get("attribute") in self.STRONG_ATTRIBUTES
        )

        # Strong shared settlement/identity evidence: check whether a human has
        # already resolved this same infrastructure before writing a narrative.
        if strong and "get_prior_review_history" not in called:
            return _action(
                "get_prior_review_history",
                {
                    "candidate_id": brief.get("candidate_id"),
                    "evidence_ids": [item["evidence_id"] for item in strong[:5]],
                },
            )
        # Weak links only: submission timing is what separates a batch from a
        # long-standing shared office, so that is the question worth asking.
        if not strong and "compare_submission_timeline" not in called:
            return _action(
                "compare_submission_timeline", {"candidate_id": brief.get("candidate_id")}
            )

        return self._conclude(brief, results, evidence, strong)

    def _conclude(self, brief, results, evidence, strong) -> str:
        seen_ids = tuple(dict.fromkeys(item["evidence_id"] for item in evidence))
        history = results.get("get_prior_review_history", {})
        outcomes = dict(history.get("outcome_counts", {}))
        legitimate_prior = sum(
            count for key, count in outcomes.items() if key.startswith("legitimate_")
        )
        ring_prior = outcomes.get("confirmed_ring", 0)
        timeline = results.get("compare_submission_timeline", {})
        span_days = float(timeline.get("span_days", 0.0) or 0.0)
        already_supplied = int(brief.get("supplied_information_count", 0) or 0)

        strong_ids = [item["evidence_id"] for item in strong][:5]
        strong_attributes = tuple(
            dict.fromkeys(str(item.get("attribute")) for item in strong)
        )
        weak_ids = [
            item["evidence_id"] for item in evidence if item not in strong
        ][:5]

        if strong and not legitimate_prior:
            recommendation = Recommendation.HUMAN_REVIEW
            attribute_names = {
                "bank_account": "a settlement account",
                "owner_pan": "an owner identifier",
                "device_fingerprint": "a device fingerprint",
            }
            shared_description = _natural_list(
                tuple(attribute_names.get(item, item.replace("_", " ")) for item in strong_attributes)
            )
            if "bank_account" in strong_attributes:
                alternative_label = "shared_settlement_or_accountant_relationship"
                missing = (
                    "authorization explaining why each merchant uses the shared settlement account",
                )
            elif "owner_pan" in strong_attributes:
                alternative_label = "disclosed_family_or_common_ownership"
                missing = (
                    "company records explaining the shared declared owner",
                )
            else:
                alternative_label = "shared_service_provider_or_kiosk"
                missing = (
                    "independent confirmation explaining the shared application device",
                )
            narrative = (
                f"Members share {shared_description}, creating a strong link "
                "between otherwise separate applications. "
                + (
                    f"{ring_prior} prior case(s) on this infrastructure were "
                    "resolved as a confirmed ring. "
                    if ring_prior
                    else "No prior human outcome exists for this infrastructure. "
                )
                + "A human reviewer should confirm whether a single controlling "
                "party is legitimate here."
            )
            requested: Tuple[InformationRequestItem, ...] = ()
        elif legitimate_prior:
            recommendation = Recommendation.CONTINUE_ONBOARDING
            alternative_label = "previously_resolved_legitimate_group"
            narrative = (
                f"{legitimate_prior} prior case(s) on this same hashed "
                "infrastructure were resolved by a human as a legitimate shared "
                "arrangement. The current links are consistent with that "
                "resolution rather than with a coordinated ring. This is a "
                "recommendation only; the deterministic policy still owns the action."
            )
            missing = ()
            requested = ()
        elif already_supplied:
            recommendation = Recommendation.HUMAN_REVIEW
            alternative_label = "shared_office_or_coworking"
            narrative = (
                "Follow-up information was supplied but the weak shared "
                "attributes remain unexplained by durable shared infrastructure. "
                "A human reviewer should make the final call."
            )
            missing = ("independent confirmation of the shared premises",)
            requested = ()
        elif span_days >= 30.0:
            recommendation = Recommendation.CONTINUE_ONBOARDING
            alternative_label = "long_standing_shared_premises"
            narrative = (
                f"Links rest on weak attributes only and the submissions span "
                f"{span_days:.1f} days, which is more consistent with durable "
                "shared premises or infrastructure than with a coordinated batch."
            )
            missing = ()
            requested = ()
        else:
            recommendation = Recommendation.REQUEST_INFORMATION
            alternative_label = "shared_office_or_coworking"
            narrative = (
                f"Links rest on weak attributes and the submissions cluster "
                f"within {span_days:.1f} days. That is consistent either with a "
                "coordinated batch or with colleagues in one office filing "
                "together. One decisive fact would separate them."
            )
            missing = ("explanation of the shared premises or network",)
            requested = (
                InformationRequestItem(
                    item_code="shared_premises_explanation",
                    description=(
                        "Explain the relationship between these applicants and "
                        "the shared address or network they submitted from."
                    ),
                ),
            )

        cited = tuple(dict.fromkeys((*strong_ids, *weak_ids))) or seen_ids[:3]
        output = InvestigatorOutput(
            recommendation=recommendation,
            ring_hypothesis=Hypothesis(
                kind=HypothesisKind.COORDINATED_RING,
                label="coordinated_merchant_ring",
                statement=(
                    "The applications were created by one coordinating party "
                    "reusing onboarding infrastructure across nominally "
                    "independent merchants."
                ),
                supporting_evidence_ids=tuple(strong_ids) or tuple(cited[:2]),
                contradicting_evidence_ids=(),
            ),
            legitimate_alternatives=(
                Hypothesis(
                    kind=HypothesisKind.LEGITIMATE_ALTERNATIVE,
                    label=alternative_label,
                    statement=(
                        "The shared attributes reflect a legitimate shared "
                        "arrangement rather than concealed common control."
                    ),
                    supporting_evidence_ids=tuple(weak_ids) or tuple(cited[:1]),
                    contradicting_evidence_ids=tuple(strong_ids),
                ),
            ),
            missing_evidence=missing,
            requested_information=requested,
            confidence=0.62 if recommendation is Recommendation.REQUEST_INFORMATION else 0.74,
            cited_evidence_ids=cited,
            narrative=narrative,
        )
        payload = json.loads(output.model_dump_json())
        payload["action"] = "conclude"
        return json.dumps(payload)


class AnthropicInvestigator:
    """The real single LLM agent, behind the same narrow provider contract."""

    name = "anthropic"

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "claude-opus-5",
        effort: str = "high",
    ) -> None:
        if not api_key:
            raise ProviderDisabled("An Anthropic API key is required.")
        from anthropic import AsyncAnthropic

        self._client = AsyncAnthropic(api_key=api_key)
        self.model = model
        self.effort = effort
        self.version = f"anthropic-{model}"

    async def propose(self, *, system, transcript, max_tokens) -> str:
        import anthropic

        try:
            response = await self._client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system,
                thinking={"type": "adaptive"},
                output_config={"effort": self.effort},
                messages=_to_messages(transcript),
            )
        except anthropic.APIError as exc:
            raise ProviderError(f"Anthropic request failed: {type(exc).__name__}") from exc
        if response.stop_reason == "refusal":
            raise ProviderError("Anthropic declined the investigation request.")
        text = "".join(
            block.text for block in response.content if block.type == "text"
        ).strip()
        if not text:
            raise ProviderError("Anthropic returned no text content.")
        return text


class BoundedInvestigator:
    """Runs one investigation under explicit step, tool, and time budgets."""

    def __init__(
        self,
        *,
        provider: InvestigatorProvider,
        budget: InvestigationBudget = InvestigationBudget(),
        validator: Optional[CitationValidator] = None,
    ) -> None:
        self.provider = provider
        self.budget = budget
        self.validator = validator or CitationValidator()

    async def investigate(
        self,
        *,
        candidate: RingCandidate,
        applications: Mapping[str, MerchantApplication],
        assessment: CandidateAssessment,
        memory: Optional[CaseMemory] = None,
        supplied_information: Sequence[Mapping[str, str]] = (),
    ) -> InvestigationResult:
        started = time.perf_counter()
        catalogue = build_evidence_catalogue(candidate, assessment)
        toolbox = InvestigationToolbox(
            candidate=candidate,
            applications=applications,
            assessment=assessment,
            catalogue=catalogue,
            memory=memory,
            supplied_information=supplied_information,
        )
        transcript: List[Dict[str, Any]] = [
            {
                "role": "user",
                "type": "case_brief",
                "content": self._case_brief(candidate, assessment, supplied_information),
            }
        ]
        observations: List[ToolObservation] = []
        disclosed: set[str] = set()
        steps = 0
        tool_calls = 0

        def finish(**kwargs) -> InvestigationResult:
            return InvestigationResult(
                candidate_id=candidate.candidate_id,
                observations=tuple(observations),
                steps_used=steps,
                tool_calls_used=tool_calls,
                duration_ms=round((time.perf_counter() - started) * 1000.0, 3),
                provider_name=self.provider.name,
                provider_version=self.provider.version,
                budget=self.budget.to_dict(),
                **kwargs,
            )

        while steps < self.budget.max_steps:
            remaining = self.budget.max_seconds - (time.perf_counter() - started)
            if remaining <= 0:
                return finish(
                    status=InvestigationStatus.FAILED, error_code="INVESTIGATION_TIMEOUT"
                )
            steps += 1
            try:
                raw = await asyncio.wait_for(
                    self.provider.propose(
                        system=SYSTEM_PROMPT,
                        transcript=tuple(transcript),
                        max_tokens=self.budget.max_output_tokens,
                    ),
                    timeout=remaining,
                )
            except ProviderDisabled:
                return finish(
                    status=InvestigationStatus.ABSTAINED, error_code="LLM_DISABLED"
                )
            except asyncio.TimeoutError:
                return finish(
                    status=InvestigationStatus.FAILED, error_code="INVESTIGATION_TIMEOUT"
                )
            except Exception:
                # Provider failures are isolated here for the same reason expert
                # failures are isolated in Phase 2: unknown is not zero risk.
                return finish(
                    status=InvestigationStatus.FAILED, error_code="PROVIDER_ERROR"
                )

            try:
                payload = json.loads(raw)
                if not isinstance(payload, dict):
                    raise ValueError("response must be a JSON object")
            except (json.JSONDecodeError, ValueError):
                return finish(
                    status=InvestigationStatus.FAILED, error_code="INVALID_JSON_RESPONSE"
                )

            action = payload.get("action")
            if action == "call_tool":
                if tool_calls >= self.budget.max_tool_calls:
                    return finish(
                        status=InvestigationStatus.FAILED,
                        error_code="TOOL_BUDGET_EXHAUSTED",
                    )
                tool_calls += 1
                observation, result = self._run_tool(toolbox, payload, steps)
                observations.append(observation)
                if observation.status is ToolCallStatus.OK:
                    disclosed.update(observation.disclosed_evidence_ids)
                    transcript.append(
                        {
                            "role": "assistant",
                            "type": "action",
                            "content": payload,
                        }
                    )
                    transcript.append(
                        {
                            "role": "user",
                            "type": "tool_result",
                            "tool": observation.tool_name,
                            "content": result,
                        }
                    )
                else:
                    transcript.append(
                        {"role": "assistant", "type": "action", "content": payload}
                    )
                    transcript.append(
                        {
                            "role": "user",
                            "type": "tool_error",
                            "tool": observation.tool_name,
                            "content": {
                                "error_code": observation.error_code,
                                "message": observation.summary,
                            },
                        }
                    )
                continue

            if action != "conclude":
                return finish(
                    status=InvestigationStatus.FAILED, error_code="UNKNOWN_ACTION"
                )

            try:
                output = InvestigatorOutput.model_validate(
                    {key: value for key, value in payload.items() if key != "action"}
                )
            except ValidationError:
                return finish(
                    status=InvestigationStatus.FAILED,
                    error_code="INVALID_OUTPUT_SCHEMA",
                )

            grounding = self.validator.validate(
                output, catalogue=catalogue, disclosed_evidence_ids=disclosed
            )
            if grounding.status is GroundingStatus.INSUFFICIENT_GROUNDING:
                return finish(
                    status=InvestigationStatus.FAILED,
                    grounding=grounding,
                    error_code=grounding.error_code or "INSUFFICIENT_GROUNDING",
                    rejected_narrative=output.narrative,
                )

            if output.recommendation is Recommendation.REQUEST_INFORMATION:
                status = InvestigationStatus.AWAITING_INFORMATION
            elif output.recommendation is Recommendation.ABSTAIN:
                status = InvestigationStatus.ABSTAINED
            else:
                status = InvestigationStatus.COMPLETED
            return finish(status=status, output=output, grounding=grounding)

        return finish(
            status=InvestigationStatus.FAILED, error_code="STEP_BUDGET_EXHAUSTED"
        )

    def _run_tool(
        self,
        toolbox: InvestigationToolbox,
        payload: Mapping[str, Any],
        step: int,
    ) -> Tuple[ToolObservation, Mapping[str, Any]]:
        tool_name = str(payload.get("tool") or "unknown")
        raw_arguments = payload.get("arguments") or {}
        arguments = (
            {str(key): _stringify(value) for key, value in raw_arguments.items()}
            if isinstance(raw_arguments, Mapping)
            else {}
        )
        started = time.perf_counter()
        try:
            result = toolbox.call(tool_name, raw_arguments if isinstance(raw_arguments, Mapping) else {})
        except ToolError as exc:
            return (
                ToolObservation(
                    step=step,
                    tool_name=tool_name if tool_name in TOOL_NAMES else "rejected_tool",
                    arguments=arguments,
                    status=ToolCallStatus.ERROR,
                    summary=exc.message,
                    error_code=exc.code,
                    duration_ms=round((time.perf_counter() - started) * 1000.0, 3),
                ),
                {},
            )
        ids = disclosed_evidence_ids(result)
        return (
            ToolObservation(
                step=step,
                tool_name=tool_name,
                arguments=arguments,
                status=ToolCallStatus.OK,
                summary=f"{tool_name} returned {len(ids)} citable evidence ID(s).",
                disclosed_evidence_ids=ids,
                duration_ms=round((time.perf_counter() - started) * 1000.0, 3),
            ),
            result,
        )

    @staticmethod
    def _case_brief(
        candidate: RingCandidate,
        assessment: CandidateAssessment,
        supplied_information: Sequence[Mapping[str, str]],
    ) -> Mapping[str, Any]:
        return {
            "candidate_id": candidate.candidate_id,
            "component_size": len(candidate.member_ids),
            "member_ids": list(candidate.member_ids),
            "primary_risk_score": assessment.primary_risk_score,
            "policy_action": assessment.action.value,
            "reason_codes": list(assessment.reason_codes),
            "available_tools": {name: TOOL_DESCRIPTIONS[name] for name in TOOL_NAMES},
            "supplied_information_count": len(supplied_information),
            "supplied_information": [dict(item) for item in supplied_information],
            "note": (
                "The primary risk score is fixed. Explain it; do not recompute it."
            ),
        }


def _action(tool: str, arguments: Mapping[str, Any]) -> str:
    return json.dumps({"action": "call_tool", "tool": tool, "arguments": dict(arguments)})


def _natural_list(items: Sequence[str]) -> str:
    values = tuple(item for item in items if item)
    if not values:
        return "shared infrastructure"
    if len(values) == 1:
        return values[0]
    return ", ".join(values[:-1]) + f" and {values[-1]}"


def _first_of_type(
    transcript: Sequence[Mapping[str, Any]], entry_type: str
) -> Optional[Mapping[str, Any]]:
    for entry in transcript:
        if entry.get("type") == entry_type:
            content = entry.get("content")
            return content if isinstance(content, Mapping) else None
    return None


def _tool_results(
    transcript: Sequence[Mapping[str, Any]],
) -> Dict[str, Mapping[str, Any]]:
    results: Dict[str, Mapping[str, Any]] = {}
    for entry in transcript:
        if entry.get("type") == "tool_result":
            content = entry.get("content")
            if isinstance(content, Mapping):
                results[str(entry.get("tool"))] = content
    return results


def _to_messages(transcript: Sequence[Mapping[str, Any]]) -> List[Dict[str, str]]:
    """Flatten the structured transcript into Anthropic message turns."""
    messages: List[Dict[str, str]] = []
    for entry in transcript:
        role = "assistant" if entry.get("role") == "assistant" else "user"
        body = json.dumps(entry.get("content", {}), sort_keys=True, default=str)
        label = entry.get("type", "message")
        tool = entry.get("tool")
        header = f"[{label}{':' + str(tool) if tool else ''}]"
        text = body if role == "assistant" else f"{header}\n{body}"
        if messages and messages[-1]["role"] == role:
            messages[-1]["content"] += "\n" + text
        else:
            messages.append({"role": role, "content": text})
    return messages


def _stringify(value: Any) -> str:
    if isinstance(value, str):
        return value[:200]
    return json.dumps(value, default=str)[:200]


def build_investigator_provider(
    *, settings: Optional[object] = None, fallback: str = "disabled"
) -> InvestigatorProvider:
    """Select a provider from configuration.

    With no API key the system does not quietly guess. It either abstains with a
    visible `LLM_DISABLED` label (the default, and what the demonstration
    shows), or runs the deterministic policy investigator when the caller has
    explicitly asked for a reproducible offline run.
    """
    if fallback not in {"disabled", "deterministic"}:
        raise ValueError("fallback must be 'disabled' or 'deterministic'")
    if settings is None:
        from ..config import get_settings

        settings = get_settings()
    api_key = getattr(settings, "anthropic_api_key", "")
    if not api_key:
        return DisabledProvider() if fallback == "disabled" else DeterministicInvestigator()
    return AnthropicInvestigator(
        api_key=api_key,
        model=getattr(settings, "investigator_model", "claude-opus-5"),
        effort=getattr(settings, "investigator_effort", "high"),
    )

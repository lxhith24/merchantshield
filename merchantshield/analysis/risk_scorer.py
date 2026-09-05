"""Risk scorer — fuses the three modules into one gated decision.

Design notes:
  * Module weights are renormalised over the modules that actually produced a
    measurement. If document analysis was skipped (no API key, no uploads), its
    weight is redistributed rather than silently scored as 0.0 "clean" — a
    missing measurement must not look like a passing one.
  * Hard rules fire before thresholds: a structurally invalid PAN or a
    settlement account shared with a rejected merchant is decisive regardless
    of the fused average.
  * Every decision carries the thresholds in force and the ranked evidence, so
    the record is auditable after the fact.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from ..config import get_settings
from ..models.document import DocumentAnalysis
from ..models.risk_assessment import (
    ClusterInfo,
    Decision,
    RiskAssessment,
    RiskSignal,
)

BASE_WEIGHTS: Dict[str, float] = {
    "document_intelligence": 0.35,
    "synthetic_identity": 0.30,
    "application_clustering": 0.35,
}

AUTO_APPROVE_THRESHOLD = 0.25
AUTO_REJECT_THRESHOLD = 0.75

# Fusion shape: how much of the final score comes from the corroborated
# weighted mean vs. the single strongest module. Weighting the max at 0.45
# means one module at 0.45+ alone is enough to deny auto-approval, which is the
# behaviour a pre-activation gate needs — silence in the other modules is
# absence of evidence, not evidence of legitimacy.
MEAN_WEIGHT = 0.55
MAX_WEIGHT = 0.45

# Any single signal at or above this severity forces a reject on its own.
DECISIVE_SIGNAL = 0.88


class RiskScorer:
    """Combines module output into a final score, decision, and explanation."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self._client = None
        if self.settings.llm_enabled:
            from anthropic import AsyncAnthropic

            self._client = AsyncAnthropic(api_key=self.settings.anthropic_api_key)

    async def score(
        self,
        application_id: str,
        document_results: Sequence[DocumentAnalysis],
        synthetic_result: Dict,
        clustering_result: Dict,
        business_name: str = "",
    ) -> RiskAssessment:
        signals: List[RiskSignal] = []

        # ---- documents -------------------------------------------------
        measured_docs = [d for d in document_results if not d.analysis_skipped]
        doc_score = max((d.document_risk_score for d in measured_docs), default=0.0)
        signals += self._document_signals(measured_docs)

        # ---- synthetic identity ---------------------------------------
        synth_score = float(synthetic_result.get("synthetic_score", 0.0))
        signals += list(synthetic_result.get("signals", []))

        # ---- clustering ------------------------------------------------
        cluster_score = float(clustering_result.get("clustering_risk_score", 0.0))
        signals += list(clustering_result.get("signals", []))
        cluster_info: Optional[ClusterInfo] = clustering_result.get("cluster_info")

        # ---- fusion over modules that actually measured ----------------
        # Two-part fusion. A plain weighted average is the wrong shape for a
        # gate: with one module silent, a serious finding in another gets
        # renormalised toward zero and auto-approves (e.g. identity 0.45 with
        # no ring links diluted to 0.21). So blend the corroborated view
        # (weighted mean) with the strongest single module, letting one loud
        # module carry an application into review on its own while multiple
        # agreeing modules still score highest.
        active: Dict[str, float] = {
            "synthetic_identity": synth_score,
            "application_clustering": cluster_score,
        }
        if measured_docs:
            active["document_intelligence"] = doc_score

        total_w = sum(BASE_WEIGHTS[m] for m in active)
        weighted_mean = sum(BASE_WEIGHTS[m] * s for m, s in active.items()) / total_w
        strongest = max(active.values())
        final = MEAN_WEIGHT * weighted_mean + MAX_WEIGHT * strongest

        # Corroboration: several independent severe tells compound.
        severe = sum(1 for s in signals if s.value >= 0.7)
        if severe >= 3:
            final = min(1.0, final * 1.18)
        elif severe == 2:
            final = min(1.0, final * 1.08)

        final = round(min(1.0, max(0.0, final)), 4)

        decision, reason = self._decide(final, signals, cluster_info)
        mode = "full" if self.settings.llm_enabled else "heuristic_only"

        assessment = RiskAssessment(
            application_id=application_id,
            document_risk_score=round(doc_score, 4),
            synthetic_identity_score=round(synth_score, 4),
            clustering_risk_score=round(cluster_score, 4),
            risk_signals=signals,
            cluster_info=cluster_info,
            final_risk_score=final,
            decision=decision,
            decision_reason=reason,
            auto_approve_threshold=AUTO_APPROVE_THRESHOLD,
            auto_reject_threshold=AUTO_REJECT_THRESHOLD,
            analysis_mode=mode,
        )
        assessment.human_readable_explanation = await self._explain(
            assessment, business_name, document_results
        )
        return assessment

    # ------------------------------------------------------------- documents
    @staticmethod
    def _document_signals(docs: Sequence[DocumentAnalysis]) -> List[RiskSignal]:
        out: List[RiskSignal] = []
        for d in docs:
            label = d.document_type.value.replace("_", " ")
            if d.tampering_detected:
                out.append(RiskSignal(
                    name=f"tampering_{d.document_type.value}",
                    value=max(d.tampering_confidence, 0.5),
                    weight=1.5,
                    explanation=(
                        f"Tampering detected in the {label}: "
                        + ("; ".join(d.tampering_reasons[:2]) or "manipulation artifacts present")
                    ),
                    source="document_intelligence",
                ))
            if d.is_template_forgery:
                out.append(RiskSignal(
                    name=f"template_forgery_{d.document_type.value}",
                    value=max(d.template_match_score, 0.5),
                    weight=1.4,
                    explanation=(
                        f"The {label} appears generated from a forgery template "
                        f"rather than issued by the authority."
                    ),
                    source="document_intelligence",
                ))
            for mismatch in d.field_mismatches[:3]:
                out.append(RiskSignal(
                    name=f"field_mismatch_{d.document_type.value}",
                    value=0.55, weight=1.2,
                    explanation=f"{label.capitalize()} disagrees with the form: {mismatch}",
                    source="document_intelligence",
                ))
        return out

    # -------------------------------------------------------------- decision
    @staticmethod
    def _decide(
        score: float,
        signals: Sequence[RiskSignal],
        cluster: Optional[ClusterInfo],
    ) -> tuple[Decision, str]:
        # Hard rules first — these are decisive on their own.
        decisive = [s for s in signals if s.value >= DECISIVE_SIGNAL]
        if decisive:
            worst = max(decisive, key=lambda s: s.value)
            return Decision.REJECTED, f"Decisive signal — {worst.explanation}"

        if cluster and cluster.cluster_size >= 8:
            return (
                Decision.REJECTED,
                f"Linked to {cluster.cluster_size - 1} other applications "
                f"(bulk-submission ring).",
            )

        if score >= AUTO_REJECT_THRESHOLD:
            return Decision.REJECTED, f"Fused risk {score:.2f} at or above reject threshold."
        if score <= AUTO_APPROVE_THRESHOLD:
            return Decision.APPROVED, f"Fused risk {score:.2f} within auto-approve band."
        return (
            Decision.UNDER_REVIEW,
            f"Fused risk {score:.2f} falls between thresholds — manual review required.",
        )

    # ----------------------------------------------------------- explanation
    async def _explain(
        self,
        a: RiskAssessment,
        business_name: str,
        documents: Sequence[DocumentAnalysis],
    ) -> str:
        deterministic = self._template_explanation(a, business_name, documents)
        if not self._client:
            return deterministic

        evidence = "\n".join(
            f"- [{s.value:.2f}] {s.name} ({s.source}): {s.explanation}"
            for s in a.top_signals
        ) or "- No risk signals fired."

        cluster_txt = "Not linked to any other application."
        if a.cluster_info and a.cluster_info.cluster_size > 1:
            cluster_txt = (
                f"Linked to {a.cluster_info.cluster_size - 1} other application(s) "
                f"via: {', '.join(a.cluster_info.shared_attributes)}."
            )

        prompt = f"""You are a fraud analyst at a payment aggregator writing the \
decision note for a merchant onboarding review queue.

Merchant: {business_name or '(unnamed)'}
Decision: {a.decision.value.upper()}
Fused risk score: {a.final_risk_score:.2f} (0 = clean, 1 = certain fraud)
Module scores — documents {a.document_risk_score:.2f}, \
synthetic identity {a.synthetic_identity_score:.2f}, \
clustering {a.clustering_risk_score:.2f}

Evidence, strongest first:
{evidence}

Clustering: {cluster_txt}

Write 2-3 sentences for the human reviewer. State the concrete reason for the \
decision, name the specific evidence that drove it, and if this is under review \
say exactly what the reviewer should verify. Do not invent evidence beyond the \
list above. Plain prose, no bullet points, no preamble."""

        try:
            resp = await self._client.messages.create(
                model=self.settings.model,
                max_tokens=320,
                messages=[{"role": "user", "content": prompt}],
            )
            text = "".join(
                b.text for b in resp.content if getattr(b, "type", "") == "text"
            ).strip()
            return text or deterministic
        except Exception:
            # Never fail an assessment because prose generation failed.
            return deterministic

    @staticmethod
    def _template_explanation(
        a: RiskAssessment,
        business_name: str,
        documents: Sequence[DocumentAnalysis],
    ) -> str:
        name = business_name or "This application"
        top = a.top_signals

        if not top:
            body = (
                "No risk signals fired across document, identity, or clustering "
                "checks."
            )
        else:
            body = " ".join(f"{s.explanation}" for s in top[:3])

        verdict = {
            Decision.APPROVED: f"{name} was auto-approved at risk {a.final_risk_score:.2f}.",
            Decision.UNDER_REVIEW: (
                f"{name} was held for manual review at risk {a.final_risk_score:.2f}."
            ),
            Decision.REJECTED: f"{name} was rejected at risk {a.final_risk_score:.2f}.",
        }[a.decision]

        skipped = [d for d in documents if d.analysis_skipped]
        caveat = ""
        if skipped:
            caveat = (
                f" Note: {len(skipped)} document(s) were not analysed "
                f"({skipped[0].skip_reason}) — this score reflects identity and "
                f"clustering evidence only."
            )
        return f"{verdict} {body}{caveat}".strip()

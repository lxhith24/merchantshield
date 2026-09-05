"""Typed-ish HTTP client for the MerchantShield review API.

The demonstration UI never imports the runtime directly. It talks to the same
REST surface a real reviewer tool would, so anything the UI can do is a thing
the API genuinely supports -- including losing an optimistic-concurrency race,
which surfaces here as `VersionConflict` rather than as a silent overwrite.
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

import httpx

DEFAULT_BASE_URL = "http://127.0.0.1:8000"


class ApiError(RuntimeError):
    """Any non-success response from the API, with the server's own detail."""

    def __init__(self, status_code: int, detail: Any) -> None:
        super().__init__(f"API returned {status_code}: {detail}")
        self.status_code = status_code
        self.detail = detail


class VersionConflict(ApiError):
    """The case moved on since it was read. Re-read and retry deliberately."""

    def __init__(self, detail: Mapping[str, Any]) -> None:
        super().__init__(409, detail.get("message", "concurrent modification"))
        self.expected_version = detail.get("expected_version")
        self.current_version = detail.get("current_version")


class MerchantShieldClient:
    """Thin, synchronous wrapper. One method per endpoint, no caching."""

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        *,
        timeout: float = 30.0,
        http_client: Optional[httpx.Client] = None,
    ) -> None:
        """`http_client` lets tests hand in a `TestClient` and skip the socket."""
        self.base_url = base_url.rstrip("/")
        self._client = http_client or httpx.Client(
            base_url=self.base_url, timeout=timeout
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "MerchantShieldClient":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # -- plumbing ---------------------------------------------------------

    def _request(self, method: str, path: str, **kwargs: Any) -> Dict[str, Any]:
        response = self._client.request(method, path, **kwargs)
        if response.status_code >= 400:
            raise _to_error(response)
        return response.json()

    def _get(self, path: str, **params: Any) -> Dict[str, Any]:
        query = {key: value for key, value in params.items() if value is not None}
        return self._request("GET", path, params=query)

    def _post(self, path: str, payload: Mapping[str, Any]) -> Dict[str, Any]:
        return self._request("POST", path, json=dict(payload))

    # -- meta -------------------------------------------------------------

    def health(self) -> Dict[str, Any]:
        return self._get("/health")

    def status(self) -> Dict[str, Any]:
        return self._get("/api/v1/meta/status")

    def vocabulary(self) -> Dict[str, Any]:
        return self._get("/api/v1/reference/review-vocabulary")

    def evaluation_report(self, dataset: str = "performance") -> Dict[str, Any]:
        return self._get("/api/v1/evaluation/report", dataset=dataset)

    # -- demonstration ----------------------------------------------------

    def scenarios(self) -> List[Dict[str, Any]]:
        return list(self._get("/api/v1/demo/scenarios")["scenarios"])

    def run_harness(self, key: str) -> Dict[str, Any]:
        return self._post(f"/api/v1/demo/harness/{key}", {})

    # -- candidates and cases --------------------------------------------

    def candidates(self, *, min_members: int = 1, limit: int = 100) -> Dict[str, Any]:
        return self._get("/api/v1/candidates", min_members=min_members, limit=limit)

    def open_case(
        self, candidate_id: str, *, idempotency_key: Optional[str] = None
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"candidate_id": candidate_id}
        if idempotency_key:
            payload["idempotency_key"] = idempotency_key
        return self._post("/api/v1/cases", payload)

    def list_cases(
        self,
        *,
        status: Optional[str] = None,
        claimed_by: Optional[str] = None,
        unclaimed_only: bool = False,
        min_risk: Optional[float] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Dict[str, Any]:
        return self._get(
            "/api/v1/cases",
            status=status,
            claimed_by=claimed_by,
            unclaimed_only=unclaimed_only or None,
            min_risk=min_risk,
            limit=limit,
            offset=offset,
        )

    def stats(self) -> Dict[str, Any]:
        return self._get("/api/v1/cases/stats")

    def case(self, case_id: str) -> Dict[str, Any]:
        return self._get(f"/api/v1/cases/{case_id}")

    def events(self, case_id: str) -> Dict[str, Any]:
        return self._get(f"/api/v1/cases/{case_id}/events")

    def claim(self, case_id: str, reviewer_id: str, version: int) -> Dict[str, Any]:
        return self._post(
            f"/api/v1/cases/{case_id}/claim",
            {"reviewer_id": reviewer_id, "expected_version": version},
        )

    def release(self, case_id: str, reviewer_id: str, version: int) -> Dict[str, Any]:
        return self._post(
            f"/api/v1/cases/{case_id}/release",
            {"reviewer_id": reviewer_id, "expected_version": version},
        )

    def supply_information(
        self,
        case_id: str,
        *,
        version: int,
        items: Sequence[Mapping[str, str]],
        supplied_by: str = "reviewer",
    ) -> Dict[str, Any]:
        return self._post(
            f"/api/v1/cases/{case_id}/information",
            {
                "expected_version": version,
                "supplied_by": supplied_by,
                "items": [dict(item) for item in items],
            },
        )

    def resolve(
        self,
        case_id: str,
        *,
        reviewer_id: str,
        version: int,
        decision: str,
        outcome: str,
        reason_codes: Sequence[str],
        notes: str = "",
    ) -> Dict[str, Any]:
        return self._post(
            f"/api/v1/cases/{case_id}/resolve",
            {
                "reviewer_id": reviewer_id,
                "expected_version": version,
                "decision": decision,
                "outcome": outcome,
                "reason_codes": list(reason_codes),
                "notes": notes,
            },
        )

    def hand_off(self, case_id: str, *, actor: str, version: int) -> Dict[str, Any]:
        return self._post(
            f"/api/v1/cases/{case_id}/handoff",
            {"actor": actor, "expected_version": version},
        )


def _to_error(response: httpx.Response) -> ApiError:
    try:
        detail = response.json().get("detail", response.text)
    except ValueError:
        detail = response.text
    if response.status_code == 409 and isinstance(detail, Mapping):
        if detail.get("error") == "concurrent_modification":
            return VersionConflict(detail)
    return ApiError(response.status_code, detail)

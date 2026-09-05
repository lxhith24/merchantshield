"""Deterministic SVG rendering of one candidate ring.

The layout is a fixed circle ordered by member ID, so the same candidate draws
identically on every run and in every recording. Edges are coloured by their
strongest shared attribute and thickened by pair strength.

This module renders only what the API already disclosed: hashed values, never
raw identifiers.
"""
from __future__ import annotations

import base64
import html
import math
from typing import Any, Iterable, List, Mapping, Optional, Sequence, Tuple

#: Attribute -> (stroke colour, human label). Ordered strongest first, which is
#: also the order the legend uses.
ATTRIBUTE_STYLE: Tuple[Tuple[str, str, str], ...] = (
    ("bank_account", "#c3a173", "Settlement account"),
    ("owner_pan", "#c09780", "Owner PAN"),
    ("device_fingerprint", "#7aafa9", "Device fingerprint"),
    ("address_hash", "#82a8bf", "Registered address"),
    ("ip_address", "#8996af", "Submission IP"),
    ("phone_series", "#b09abe", "Phone series"),
    ("email_domain", "#9aa69d", "Email domain"),
)

_COLOUR = {attribute: colour for attribute, colour, _ in ATTRIBUTE_STYLE}
_LABEL = {attribute: label for attribute, _, label in ATTRIBUTE_STYLE}
_RANK = {attribute: index for index, (attribute, _, _) in enumerate(ATTRIBUTE_STYLE)}

NODE_RADIUS = 26.0
_FALLBACK_COLOUR = "#64748b"


def attribute_colour(attribute: str) -> str:
    return _COLOUR.get(attribute, _FALLBACK_COLOUR)


def attribute_label(attribute: str) -> str:
    return _LABEL.get(attribute, attribute.replace("_", " ").title())


def _positions(
    member_ids: Sequence[str], width: float, height: float
) -> Mapping[str, Tuple[float, float]]:
    """Fixed circular layout. One member sits in the centre."""
    centre_x, centre_y = width / 2.0, height / 2.0
    if len(member_ids) == 1:
        return {member_ids[0]: (centre_x, centre_y)}
    radius = min(width, height) / 2.0 - NODE_RADIUS - 46.0
    step = 2.0 * math.pi / len(member_ids)
    return {
        member_id: (
            centre_x + radius * math.sin(index * step),
            centre_y - radius * math.cos(index * step),
        )
        for index, member_id in enumerate(member_ids)
    }


def _strongest_attribute(attributes: Iterable[str]) -> Optional[str]:
    ranked = sorted(attributes, key=lambda item: _RANK.get(item, len(_RANK)))
    return ranked[0] if ranked else None


def _pair_key(left: str, right: str) -> Tuple[str, str]:
    return (left, right) if left <= right else (right, left)


def _short(member_id: str) -> str:
    """`syn-0042` -> `MS-0042`; anything else is truncated, never invented."""
    tail = member_id.rsplit("-", 1)[-1]
    return "MS-" + (tail if len(tail) <= 6 else tail[:6])


def _display_name(member: Optional[Mapping[str, Any]], member_id: str) -> str:
    if not member or not member.get("business_name"):
        return member_id
    name = str(member["business_name"])
    return name if len(name) <= 22 else name[:21] + "…"


def render_ring_svg(
    candidate: Mapping[str, Any],
    *,
    members: Sequence[Mapping[str, Any]] = (),
    width: int = 720,
    height: int = 460,
    highlight_evidence_ids: Sequence[str] = (),
) -> str:
    """Return a standalone SVG for one candidate group.

    `highlight_evidence_ids` dims every edge the investigator did not cite, so
    a reviewer can see exactly which links the narrative actually rests on.
    """
    member_ids: List[str] = [str(item) for item in candidate.get("member_ids", ())]
    if not member_ids:
        return _empty_svg(width, height, "No members in this candidate group.")

    positions = _positions(member_ids, float(width), float(height))
    by_id = {str(item.get("member_id")): item for item in members}
    highlighted = {str(item) for item in highlight_evidence_ids}

    strengths = {
        _pair_key(str(row["left_id"]), str(row["right_id"])): float(row["strength"])
        for row in candidate.get("pair_strengths", ())
    }

    pair_attributes: dict[Tuple[str, str], set[str]] = {}
    pair_evidence: dict[Tuple[str, str], set[str]] = {}
    for edge in candidate.get("evidence", ()):
        key = _pair_key(str(edge["left_id"]), str(edge["right_id"]))
        pair_attributes.setdefault(key, set()).add(str(edge["attribute"]))
        pair_evidence.setdefault(key, set()).add(str(edge["evidence_id"]))

    parts: List[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="100%" height="{height}" role="img" '
        f'aria-label="Evidence links between {len(member_ids)} applications">',
        f'<rect width="{width}" height="{height}" rx="20" fill="#101616"/>',
    ]

    for key in sorted(pair_attributes):
        left, right = key
        if left not in positions or right not in positions:
            continue
        x1, y1 = positions[left]
        x2, y2 = positions[right]
        attribute = _strongest_attribute(pair_attributes[key]) or ""
        strength = strengths.get(key, 0.0)
        stroke_width = round(1.0 + 2.0 * max(0.0, min(1.0, strength)), 2)
        cited = bool(highlighted & pair_evidence.get(key, set()))
        opacity = 0.95 if (not highlighted or cited) else 0.18
        title = (
            f"{', '.join(attribute_label(item) for item in sorted(pair_attributes[key]))}"
            f" — strength {strength:.2f}"
        )
        parts.append(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="{attribute_colour(attribute)}" stroke-width="{stroke_width}" '
            f'stroke-opacity="{opacity}" stroke-linecap="round">'
            f"<title>{html.escape(title)}</title></line>"
        )

    for member_id in member_ids:
        x, y = positions[member_id]
        member = by_id.get(member_id)
        name = _display_name(member, member_id)
        parts.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{NODE_RADIUS}" fill="#1e2b29" '
            'stroke="#57756e" stroke-width="1.2"/>'
            f'<text x="{x:.1f}" y="{y + 4:.1f}" text-anchor="middle" '
            'font-family="Times New Roman, Times, serif" '
            'font-size="10.5" font-weight="600" '
            f'fill="#e5ece8">{html.escape(_short(member_id))}</text>'
            f'<rect x="{x - 72:.1f}" y="{y + NODE_RADIUS + 5:.1f}" '
            'width="144" height="22" rx="6" fill="#1e2b29" stroke="#364b45"/>'
            f'<text x="{x:.1f}" y="{y + NODE_RADIUS + 20:.1f}" text-anchor="middle" '
            'font-family="Times New Roman, Times, serif" '
            'font-size="10.5" fill="#c1ccc7">'
            f"{html.escape(name)}</text>"
        )

    present = sorted(
        {attribute for values in pair_attributes.values() for attribute in values},
        key=lambda item: _RANK.get(item, len(_RANK)),
    )
    parts.append(_legend(present, width, height))
    if len(member_ids) == 1:
        parts.append(
            f'<text x="{width / 2:.0f}" y="{height - 30}" text-anchor="middle" '
            'font-family="Times New Roman, Times, serif" font-size="12" fill="#a5b5ae">'
            "No linked applications: nothing to corroborate, nothing to clear."
            "</text>"
        )
    parts.append("</svg>")
    return "".join(parts)


def svg_data_uri(svg: str) -> str:
    """Encode a locally generated SVG for Streamlit's image renderer."""
    encoded = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    return f"data:image/svg+xml;base64,{encoded}"


def _legend(attributes: Sequence[str], width: int, height: int) -> str:
    if not attributes:
        return ""
    parts = ['<g font-family="Times New Roman, Times, serif" font-size="11" fill="#b4c0bb">']
    x = 16.0
    y = float(height - 12)
    for attribute in attributes:
        parts.append(
            f'<line x1="{x:.0f}" y1="{y - 4:.0f}" x2="{x + 18:.0f}" y2="{y - 4:.0f}" '
            f'stroke="{attribute_colour(attribute)}" stroke-width="3.5" '
            'stroke-linecap="round"/>'
            f'<text x="{x + 24:.0f}" y="{y:.0f}">'
            f"{html.escape(attribute_label(attribute))}</text>"
        )
        x += 34.0 + 7.0 * len(attribute_label(attribute))
        if x > width - 90:
            break
    parts.append("</g>")
    return "".join(parts)


def _empty_svg(width: int, height: int, message: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="100%" height="{height}">'
        f'<rect width="{width}" height="{height}" rx="14" fill="#101616" '
        'stroke="#364b45"/>'
        f'<text x="{width / 2:.0f}" y="{height / 2:.0f}" text-anchor="middle" '
        'font-family="Times New Roman, Times, serif" font-size="13" fill="#a5b5ae">'
        f"{html.escape(message)}</text></svg>"
    )

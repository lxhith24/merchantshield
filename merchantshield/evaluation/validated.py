"""Load an evaluated challenger from safe JSON/data; never load pickle files."""
import hashlib
import json
from pathlib import Path

from .dataset import load_jsonl
from .performance import MerchantPeerModel, VERSION, extract_features


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


async def load_validated_model(directory: Path):
    plan = json.loads((directory / "performance_plan.json").read_text())
    report = json.loads((directory / "performance_report.json").read_text())
    lock = plan.pop("lock_hash")
    if _digest(plan) != lock or report["manifest"]["lock_hash"] != lock or plan["version"] != VERSION:
        raise ValueError("validated model manifest does not match")
    rows = load_jsonl(directory / "performance_merchants.jsonl")
    if _digest([row.to_dict() for row in rows]) != plan["dataset_hash"]:
        raise ValueError("validated training data changed")
    ids = plan["ids"]
    if set(ids["train"]) & (set(ids["validation"]) | set(ids["test"])):
        raise ValueError("training IDs overlap held-out data")
    by_id = {row.example_id: row for row in rows}
    train = [by_id[key] for key in ids["train"]]
    features = await extract_features({row.example_id: row.application for row in train})
    model = MerchantPeerModel(**plan["selected"], seed=plan["seed"])
    model.fit(features, {row.example_id: row.is_shell for row in train})
    return model, plan["thresholds"]["routed_moe"]["threshold"]


async def load_relationship_model(directory: Path):
    """Shadow-only challenger; verify its sealed plan/code/data before fitting."""
    from scripts.recover_hard_rings import load_locked, feature_batch
    from .relationships import RelationshipRiskModel

    plan, partitions = await load_locked(directory, require_report=True)
    train = partitions["train"]
    model = RelationshipRiskModel(**plan["selected"], seed=plan["seed"])
    model.fit(await feature_batch(train), {row.example_id: row.is_shell for row in train})
    return model, plan["thresholds"]["routed_moe"]["threshold"]

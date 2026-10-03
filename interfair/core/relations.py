from __future__ import annotations
from pathlib import Path as _ArtifactPath
_PACKAGE = _ArtifactPath(__file__).resolve().parents[1]

import argparse
import hashlib
import json
import math
import os
import pickle
import random
import time
import urllib.request
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from torch import nn


ADULT_URLS = {
    "adult.data": "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
    "adult.test": "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test",
}

BANK_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip"
DEFAULT_CREDIT_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls"
COMPAS_URL = "https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv"

ADULT_COLUMNS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education_num",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
    "native_country",
    "income",
]

MODEL_LABELS = {
    "erm": "ERM",
    "reweight": "Reweighted ERM",
    "blind": "Sensitive feature removed",
    "reweight_blind": "Reweighted sensitive feature removed",
    "proxy_score_only": "Score-gap steering",
    "no_interaction": "No-interaction scaffold",
    "proxy_effect": "Proxy-effect steering",
    "proxy_reweight": "Reweighted proxy-effect steering",
    "topology_main": "Main-effect steering",
    "topology_single": "Main + one interaction",
    "topology_exhaustive": "Exhaustive interaction steering",
    "topology_wrong": "Wrong interaction steering",
    "proxy_blind": "Sensitive removed + proxy steering",
    "proxy_score_only_blind": "Sensitive removed + score-gap steering",
    "proxy_reweight_blind": "Reweighted sensitive removed + proxy steering",
    "proxy_exhaustive": "Direct + proxy interaction steering",
    "direct_spec_boundary": "Direct full + boundary repair",
    "direct_spec_full": "Direct full specification repair",
    "direct_spec_full_higher_order": "Direct full + higher-order specification repair",
    "direct_spec_boundary_wrong_effect": "Direct wrong boundary specification",
    "direct_spec_wrong_effect": "Direct wrong gap-amplifying specification",
    "spec_score_blind": "Spec S1: sensitive removed + score-gap repair",
    "spec_proxy_blind": "Spec S2: + proxy-effect repair",
    "spec_boundary_blind": "Spec S3: + boundary-local repair",
    "spec_slice_blind": "Spec S4: + slice-local repair",
    "spec_full_blind": "Spec S5: full fairness specification repair",
    "spec_full_higher_order_blind": "Spec S6: full + higher-order fairness repair",
    "spec_slice_reweight_blind": "Reweighted Spec S4: + slice-local repair",
    "spec_full_reweight_blind": "Reweighted Spec S5: full fairness specification repair",
    "spec_residual_blind": "Spec S5: residual-selected repair",
    "spec_full_placebo_blind": "Spec placebo: full repair on shuffled groups",
    "spec_full_score_only_blind": "Spec ablation: full score-only repair",
    "spec_full_wrong_effect_blind": "Spec ablation: wrong gap-amplifying repair",
    "mobius_spec_full": "Mobius/ANOVA full fairness specification repair",
    "mobius_spec_full_higher_order": "Mobius/ANOVA full + higher-order fairness repair",
    "adv": "Adversarial debiasing",
    "adv_reweight": "Reweighted adversarial debiasing",
    "adv_blind": "Sensitive removed + adversarial debiasing",
    "adv_reweight_blind": "Reweighted sensitive removed + adversarial debiasing",
}

DIRECT_WRONG_MODELS = {"topology_wrong", "direct_spec_boundary_wrong_effect", "direct_spec_wrong_effect"}
DIRECT_TOPOLOGY_MODELS = {
    "topology_main",
    "topology_single",
    "topology_exhaustive",
    "topology_wrong",
    "proxy_exhaustive",
    "direct_spec_boundary",
    "direct_spec_boundary_wrong_effect",
    "direct_spec_full",
    "direct_spec_full_higher_order",
    "direct_spec_wrong_effect",
}
PROXY_STEERING_MODELS = {
    "proxy_score_only",
    "proxy_effect",
    "proxy_reweight",
    "proxy_blind",
    "proxy_score_only_blind",
    "proxy_reweight_blind",
    "proxy_exhaustive",
}
SPEC_LADDER_MODELS = {
    "no_interaction",
    "direct_spec_boundary",
    "direct_spec_boundary_wrong_effect",
    "direct_spec_full",
    "direct_spec_full_higher_order",
    "direct_spec_wrong_effect",
    "spec_score_blind",
    "spec_proxy_blind",
    "spec_boundary_blind",
    "spec_slice_blind",
    "spec_full_blind",
    "spec_full_higher_order_blind",
    "spec_slice_reweight_blind",
    "spec_full_reweight_blind",
    "spec_residual_blind",
    "spec_full_placebo_blind",
    "spec_full_score_only_blind",
    "spec_full_wrong_effect_blind",
    "mobius_spec_full",
    "mobius_spec_full_higher_order",
}
SPEC_BOUNDARY_MODELS = {
    "no_interaction",
    "direct_spec_boundary",
    "direct_spec_boundary_wrong_effect",
    "direct_spec_full",
    "direct_spec_full_higher_order",
    "direct_spec_wrong_effect",
    "spec_boundary_blind",
    "spec_full_blind",
    "spec_full_higher_order_blind",
    "spec_full_reweight_blind",
    "spec_residual_blind",
    "spec_full_placebo_blind",
    "spec_full_score_only_blind",
    "spec_full_wrong_effect_blind",
    "mobius_spec_full",
    "mobius_spec_full_higher_order",
}
SPEC_SLICE_MODELS = {
    "direct_spec_full",
    "direct_spec_full_higher_order",
    "direct_spec_wrong_effect",
    "spec_slice_blind",
    "spec_full_blind",
    "spec_full_higher_order_blind",
    "spec_slice_reweight_blind",
    "spec_full_reweight_blind",
    "spec_residual_blind",
    "spec_full_placebo_blind",
    "spec_full_score_only_blind",
    "spec_full_wrong_effect_blind",
    "mobius_spec_full",
    "mobius_spec_full_higher_order",
}
SPEC_RESIDUAL_MODELS = {"spec_residual_blind"}
PLACEBO_SPEC_MODELS = {"spec_full_placebo_blind"}
SPEC_FULL_SCORE_ONLY_MODELS = {"spec_full_score_only_blind"}
NO_INTERACTION_MODELS = {"no_interaction"}
SPEC_WRONG_EFFECT_MODELS = {"spec_full_wrong_effect_blind", "direct_spec_boundary_wrong_effect", "direct_spec_wrong_effect"}
SPEC_HIGHER_ORDER_MODELS = {"direct_spec_full_higher_order", "spec_full_higher_order_blind", "mobius_spec_full_higher_order"}
FAIRNESS_MOBIUS_MODELS = {"mobius_spec_full", "mobius_spec_full_higher_order"}
PROXY_SCORE_ONLY_MODELS = {"proxy_score_only", "proxy_score_only_blind", "spec_score_blind"}
ADVERSARIAL_MODELS = {"adv", "adv_reweight", "adv_blind", "adv_reweight_blind"}
ORDINARY_BASELINE_SELECTION_MODELS = {
    "erm",
    "reweight",
    "blind",
    "reweight_blind",
    "adv",
    "adv_reweight",
    "adv_blind",
    "adv_reweight_blind",
}
SCORE_CONTROL_SELECTION_MODELS = PROXY_SCORE_ONLY_MODELS | SPEC_FULL_SCORE_ONLY_MODELS | NO_INTERACTION_MODELS
WRONG_CONTROL_SELECTION_MODELS = DIRECT_WRONG_MODELS | SPEC_WRONG_EFFECT_MODELS
BEHAVIOR_ONLY_SELECTION_MODELS = {
    "erm",
    "reweight",
    "blind",
    "reweight_blind",
    "adv",
    "adv_reweight",
    "adv_blind",
    "adv_reweight_blind",
    "topology_wrong",
    "direct_spec_wrong_effect",
    "proxy_score_only",
    "proxy_score_only_blind",
    "spec_score_blind",
    "spec_full_score_only_blind",
    "spec_full_wrong_effect_blind",
    "spec_full_placebo_blind",
}
SENSITIVE_REMOVED_MODELS = {
    "blind",
    "reweight_blind",
    "proxy_blind",
    "proxy_score_only_blind",
    "proxy_reweight_blind",
    "spec_score_blind",
    "spec_proxy_blind",
    "spec_boundary_blind",
    "spec_slice_blind",
    "spec_full_blind",
    "spec_full_higher_order_blind",
    "spec_slice_reweight_blind",
    "spec_full_reweight_blind",
    "spec_residual_blind",
    "spec_full_placebo_blind",
    "spec_full_score_only_blind",
    "spec_full_wrong_effect_blind",
    "adv_blind",
    "adv_reweight_blind",
}
REWEIGHTED_MODELS = {
    "reweight",
    "reweight_blind",
    "proxy_reweight",
    "proxy_reweight_blind",
    "spec_slice_reweight_blind",
    "spec_full_reweight_blind",
    "adv_reweight",
    "adv_reweight_blind",
}
INTERACTION_STEERING_MODELS = DIRECT_TOPOLOGY_MODELS | PROXY_STEERING_MODELS | SPEC_LADDER_MODELS


@dataclass
class AxisSpec:
    name: str
    feature: str
    column: str
    low: float
    high: float
    low_raw: float
    high_raw: float
    missing_column: str | None = None


@dataclass
class DatasetSpec:
    name: str
    sensitive_name: str
    sensitive_positive: str
    sensitive_negative: str
    label_name: str
    favorable_label: str
    input_columns: list[str]
    sensitive_column: str
    continuous_columns: list[str]
    categorical_columns: list[str]
    axes: list[AxisSpec]
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    source_summary: dict[str, object]


@dataclass
class PreparedData:
    name: str
    x: np.ndarray
    y: np.ndarray
    s: np.ndarray
    spec: DatasetSpec
    s_placebo: np.ndarray | None = None


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, depth: int, dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current = input_dim
        for _ in range(depth):
            layers.append(nn.Linear(current, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            current = hidden_dim
        layers.append(nn.Linear(current, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        for layer in list(self.net.children())[:-1]:
            x = layer(x)
        return x


class MobiusFairnessHead(nn.Module):
    """A guarded ANOVA/Mobius logit head over the sensitive toggle and audit axes."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        depth: int,
        dropout: float,
        sensitive_col: int,
        axis_cols: list[int],
        context_mask_cols: list[int],
        axis_thresholds: list[float],
        axis_scales: list[float],
        toggle_mode: str,
    ) -> None:
        super().__init__()
        self.sensitive_col = int(sensitive_col)
        self.axis_cols = [int(col) for col in axis_cols]
        self.toggle_mode = toggle_mode
        n_toggles = 1 + len(axis_cols)
        n_terms = 2**n_toggles

        layers: list[nn.Module] = []
        current = input_dim
        for _ in range(depth):
            layers.append(nn.Linear(current, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            current = hidden_dim
        self.context_encoder = nn.Sequential(*layers)
        self.coefficient_head = nn.Linear(current, n_terms)

        context_mask = torch.ones(input_dim, dtype=torch.float32)
        for col in sorted(set(context_mask_cols)):
            if 0 <= col < input_dim:
                context_mask[col] = 0.0
        self.register_buffer("context_mask", context_mask)
        self.register_buffer("axis_thresholds", torch.tensor(axis_thresholds, dtype=torch.float32))
        self.register_buffer("axis_scales", torch.tensor(axis_scales, dtype=torch.float32))
        masks = torch.zeros(n_terms, n_toggles, dtype=torch.float32)
        for term in range(n_terms):
            for bit in range(n_toggles):
                if term & (1 << bit):
                    masks[term, bit] = 1.0
        self.register_buffer("term_masks", masks)

    def context_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.context_encoder(x * self.context_mask)

    def toggles(self, x: torch.Tensor) -> torch.Tensor:
        s_toggle = torch.clamp(x[:, self.sensitive_col], 0.0, 1.0).unsqueeze(1)
        if not self.axis_cols:
            return s_toggle
        axis_values = x[:, self.axis_cols]
        if self.toggle_mode == "soft":
            axis_toggles = torch.sigmoid((axis_values - self.axis_thresholds) / torch.clamp(self.axis_scales, min=1e-4))
        else:
            axis_toggles = (axis_values >= self.axis_thresholds).to(axis_values.dtype)
        return torch.cat([s_toggle, axis_toggles], dim=1)

    def mobius_terms(self, x: torch.Tensor) -> torch.Tensor:
        toggles = self.toggles(x)
        selected = torch.where(
            self.term_masks.unsqueeze(0) > 0.5,
            toggles.unsqueeze(1),
            torch.ones_like(toggles).unsqueeze(1),
        )
        return selected.prod(dim=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        coefficients = self.coefficient_head(self.context_features(x))
        terms = self.mobius_terms(x)
        return (coefficients * terms).sum(dim=1)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.context_features(x)


class Adversary(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx: object, x: torch.Tensor, scale: float) -> torch.Tensor:
        ctx.scale = scale
        return x.view_as(x)

    @staticmethod
    def backward(ctx: object, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        return -ctx.scale * grad_output, None


def gradient_reverse(x: torch.Tensor, scale: float) -> torch.Tensor:
    return GradientReverse.apply(x, scale)


class PostprocessedModel:
    def __init__(self, base: nn.Module, thresholds: dict[int, float]) -> None:
        self.base = base
        self.thresholds = thresholds

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fairness interaction-debugging experiments.")
    parser.add_argument("--dataset", choices=["adult", "acs_employment", "bank", "compas", "default_credit", "hmda"], default="hmda")
    parser.add_argument("--adult-dir", type=Path, default=Path("treatment_effect/data/adult"))
    parser.add_argument(
        "--adult-axis-preset",
        choices=["default", "gender_proxy", "gender_proxy_no_sparse"],
        default="default",
        help=(
            "Adult interaction axes. default preserves the original continuous-axis recipe; "
            "gender_proxy adds explicit binary marital/relationship/occupation proxy axes; "
            "gender_proxy_no_sparse also removes sparse capital-gain/loss axes from the interaction target."
        ),
    )
    parser.add_argument("--acs-root", type=Path, default=Path("data/raw/acs"))
    parser.add_argument("--acs-year", default="2018")
    parser.add_argument("--acs-horizon", choices=["1-Year", "5-Year"], default="1-Year")
    parser.add_argument("--acs-states", nargs="+", default=["MD"])
    parser.add_argument("--acs-density", type=float, default=1.0)
    parser.add_argument("--bank-dir", type=Path, default=Path("data/raw/bank"))
    parser.add_argument("--compas-path", type=Path, default=Path("data/raw/compas/compas-scores-two-years.csv"))
    parser.add_argument("--default-credit-path", type=Path, default=Path("data/raw/default_credit/default_of_credit_card_clients.xls"))
    parser.add_argument("--hmda-path", type=Path, default=Path("data/raw/hmda_2019_md.csv"))
    parser.add_argument("--hmda-url", default="https://ffiec.cfpb.gov/v2/data-browser-api/view/csv?years=2019&states=MD")
    parser.add_argument(
        "--prepared-cache-dir",
        type=Path,
        default=None,
        help="Optional cache for preprocessed arrays/splits keyed by dataset, seed, and data config.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("retrained/seed0"))
    parser.add_argument("--models", nargs="+", default=["erm", "reweight", "topology_main", "topology_single", "topology_exhaustive"], choices=list(MODEL_LABELS))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-threads", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--anchor-batch-size", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=7e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lambda-main", type=float, default=0.6)
    parser.add_argument("--lambda-pair", type=float, default=1.0)
    parser.add_argument("--lambda-tail", type=float, default=0.25)
    parser.add_argument("--lambda-proxy-score", type=float, default=0.20)
    parser.add_argument("--lambda-proxy-pair", type=float, default=0.40)
    parser.add_argument("--lambda-boundary-score", type=float, default=0.20)
    parser.add_argument("--lambda-boundary-pair", type=float, default=0.30)
    parser.add_argument("--lambda-slice-score", type=float, default=0.10)
    parser.add_argument("--lambda-slice-pair", type=float, default=0.20)
    parser.add_argument("--lambda-higher-order-pair", type=float, default=0.10)
    parser.add_argument("--lambda-higher-order-boundary-pair", type=float, default=0.10)
    parser.add_argument("--higher-order-top-frac", type=float, default=0.35)
    parser.add_argument("--wrong-effect-margin", type=float, default=0.35)
    parser.add_argument("--lambda-adv", type=float, default=0.10)
    parser.add_argument("--adv-hidden-dim", type=int, default=64)
    parser.add_argument("--tail-frac", type=float, default=0.10)
    parser.add_argument("--min-proxy-cell-size", type=int, default=8)
    parser.add_argument("--boundary-band", type=float, default=0.18)
    parser.add_argument("--boundary-eval-margin", type=float, default=0.15)
    parser.add_argument("--slice-top-frac", type=float, default=0.50)
    parser.add_argument("--residual-top-frac", type=float, default=0.35)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--selector-fairness-weight", type=float, default=0.15)
    parser.add_argument("--selector-boundary-weight", type=float, default=0.10)
    parser.add_argument("--selector-slice-weight", type=float, default=0.10)
    parser.add_argument("--selector-higher-order-weight", type=float, default=0.05)
    parser.add_argument("--selector-mode", choices=["task_mechanism", "behavior_proxy", "behavior_direct", "behavior_only"], default="task_mechanism")
    parser.add_argument(
        "--baseline-selection",
        choices=["endpoint_fairness", "utility"],
        default="utility",
        help=(
            "Checkpoint policy for ordinary baselines, score-only controls, and wrong-spec controls. "
            "endpoint_fairness preserves legacy AOD/EOD/DP-based selection; utility selects by validation task utility."
        ),
    )
    parser.add_argument(
        "--repair-selection",
        choices=["selector_mode", "utility_mechanism"],
        default="utility_mechanism",
        help=(
            "Checkpoint policy for correct interaction-repair rows. selector_mode preserves legacy behavior+mechanism "
            "selection when --selector-mode requests it; utility_mechanism uses validation task utility plus the "
            "intended mechanism residual."
        ),
    )
    parser.add_argument("--utility-selector-metric", choices=["bce", "auc", "accuracy"], default="bce")
    parser.add_argument("--selector-auc-weight", type=float, default=0.02)
    parser.add_argument("--selector-eod-weight", type=float, default=0.0)
    parser.add_argument("--selector-dp-weight", type=float, default=0.0)
    parser.add_argument("--selector-min-auc", type=float, default=0.0)
    parser.add_argument("--selector-acc-weight", type=float, default=0.0)
    parser.add_argument("--selector-min-acc", type=float, default=0.0)
    parser.add_argument(
        "--selection-min-delta",
        type=float,
        default=0.0,
        help="Minimum validation-score improvement required to reset patience.",
    )
    parser.add_argument("--threshold-grid-size", type=int, default=41)
    parser.add_argument("--fairness-mobius-toggle-mode", choices=["hard", "soft"], default="hard")
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.20)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--report-name", default="interaction_report.md")
    parser.add_argument(
        "--no-save-artifacts",
        dest="save_artifacts",
        action="store_false",
        help="Disable saving selected model checkpoints and artifact manifests.",
    )
    parser.add_argument(
        "--artifact-dir-name",
        default="selected_artifacts",
        help="Directory name under each seed directory for selected model artifacts.",
    )
    parser.set_defaults(save_artifacts=True)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def to_jsonable(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return value


def artifact_args_payload(args: argparse.Namespace) -> dict[str, object]:
    return {key: to_jsonable(value) for key, value in vars(args).items()}


def sigmoid_np(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -40.0, 40.0)))


def as_float_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.replace("%", "", regex=False).str.strip(), errors="coerce")


def parse_dti_series(series: pd.Series) -> pd.Series:
    text = series.astype(str).str.strip()
    out = pd.to_numeric(text.str.replace("%", "", regex=False), errors="coerce")
    patterns = {
        "<20%": 19.0,
        "20%-<30%": 25.0,
        "30%-<36%": 33.0,
        "50%-60%": 55.0,
        ">60%": 65.0,
    }
    for key, value in patterns.items():
        out = out.mask(text == key, value)
    return out


def download_adult_data(adult_dir: Path) -> None:
    adult_dir.mkdir(parents=True, exist_ok=True)
    for filename, url in ADULT_URLS.items():
        path = adult_dir / filename
        if not path.exists():
            urllib.request.urlretrieve(url, path)


def download_bank_data(bank_dir: Path) -> None:
    bank_dir.mkdir(parents=True, exist_ok=True)
    bank_full = bank_dir / "bank-full.csv"
    if bank_full.exists():
        return
    archive = bank_dir / "bank.zip"
    if not archive.exists():
        urllib.request.urlretrieve(BANK_URL, archive)
    with zipfile.ZipFile(archive) as handle:
        for member in handle.namelist():
            if member.endswith("bank-full.csv"):
                handle.extract(member, bank_dir)
                extracted = bank_dir / member
                if extracted != bank_full:
                    extracted.replace(bank_full)
                break
        else:
            raise FileNotFoundError("bank-full.csv not found in downloaded Bank Marketing archive")


def download_default_credit(default_credit_path: Path) -> None:
    default_credit_path.parent.mkdir(parents=True, exist_ok=True)
    if not default_credit_path.exists():
        urllib.request.urlretrieve(DEFAULT_CREDIT_URL, default_credit_path)


def download_compas(compas_path: Path) -> None:
    compas_path.parent.mkdir(parents=True, exist_ok=True)
    if not compas_path.exists():
        urllib.request.urlretrieve(COMPAS_URL, compas_path)


def download_hmda(hmda_path: Path, url: str) -> None:
    hmda_path.parent.mkdir(parents=True, exist_ok=True)
    if not hmda_path.exists():
        print(f"Downloading HMDA data from {url} -> {hmda_path}", flush=True)
        tmp = hmda_path.with_suffix(hmda_path.suffix + ".tmp")
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(hmda_path)


def stratified_indices(y: np.ndarray, s: np.ndarray, rng: np.random.Generator, val_frac: float, test_frac: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = (s.astype(np.int64) * 2 + y.astype(np.int64)).astype(np.int64)
    train: list[int] = []
    val: list[int] = []
    test: list[int] = []
    for label in sorted(np.unique(labels).tolist()):
        idx = np.flatnonzero(labels == label)
        rng.shuffle(idx)
        n = len(idx)
        n_test = max(1, int(round(n * test_frac)))
        n_val = max(1, int(round(n * val_frac)))
        n_test = min(n_test, max(0, n - 2))
        n_val = min(n_val, max(0, n - n_test - 1))
        test.extend(idx[:n_test].tolist())
        val.extend(idx[n_test : n_test + n_val].tolist())
        train.extend(idx[n_test + n_val :].tolist())
    train_arr = np.asarray(train, dtype=np.int64)
    val_arr = np.asarray(val, dtype=np.int64)
    test_arr = np.asarray(test, dtype=np.int64)
    rng.shuffle(train_arr)
    rng.shuffle(val_arr)
    rng.shuffle(test_arr)
    return train_arr, val_arr, test_arr


def label_stratified_placebo_sensitive(s: np.ndarray, y: np.ndarray, seed: int) -> np.ndarray:
    """Create a fixed fake sensitive vector while preserving each label cell."""
    rng = np.random.default_rng(seed + 104729)
    placebo = np.empty_like(s, dtype=np.float32)
    for label in sorted(np.unique(y.astype(np.int64)).tolist()):
        idx = np.flatnonzero(y.astype(np.int64) == int(label))
        values = s[idx].astype(np.float32).copy()
        rng.shuffle(values)
        placebo[idx] = values
    return placebo


def prepare_tabular_frame(
    frame: pd.DataFrame,
    y: np.ndarray,
    s: np.ndarray,
    sensitive_column: str,
    continuous: list[str],
    categorical: list[str],
    axis_features: list[str],
    train_idx: np.ndarray,
    binary_axis_features: Iterable[str] | None = None,
) -> tuple[np.ndarray, list[str], list[AxisSpec], dict[str, object]]:
    parts: list[pd.DataFrame] = []
    input_columns: list[str] = []
    binary_axis_set = set(binary_axis_features or [])

    s_frame = pd.DataFrame({sensitive_column: s.astype(np.float32)})
    parts.append(s_frame)
    input_columns.append(sensitive_column)

    axis_specs: list[AxisSpec] = []
    continuous_summary: dict[str, object] = {}
    for col in continuous:
        raw = as_float_series(frame[col]) if col != "debt_to_income_ratio" else parse_dti_series(frame[col])
        train_raw = raw.iloc[train_idx]
        median = float(train_raw.median())
        q05 = float(train_raw.quantile(0.05))
        q10 = float(train_raw.quantile(0.10))
        q25 = float(train_raw.quantile(0.25))
        q75 = float(train_raw.quantile(0.75))
        q90 = float(train_raw.quantile(0.90))
        q95 = float(train_raw.quantile(0.95))
        q97 = float(train_raw.quantile(0.97))
        q99 = float(train_raw.quantile(0.99))
        raw_min = float(train_raw.min())
        raw_max = float(train_raw.max())
        scale = float(q75 - q25)
        if scale < 1e-6:
            spread = float(q90 - q10)
            std = float(train_raw.std()) if not train_raw.dropna().empty else 0.0
            scale = max(spread, std, 1.0)
        filled = raw.fillna(median)
        z = ((filled - median) / scale).astype(np.float32)
        z_col = f"{col}__z"
        miss_col = f"{col}__missing"
        parts.append(pd.DataFrame({z_col: z.to_numpy(dtype=np.float32), miss_col: raw.isna().astype(np.float32).to_numpy()}))
        input_columns.extend([z_col, miss_col])
        continuous_summary[col] = {
            "median": median,
            "q05": q05,
            "q10": q10,
            "q25": q25,
            "q75": q75,
            "q90": q90,
            "q95": q95,
            "q97": q97,
            "q99": q99,
            "min": raw_min,
            "max": raw_max,
            "missing_frac": float(raw.isna().mean()),
        }
        if col in axis_features:
            if col in binary_axis_set:
                axis_low = 0.0
                axis_high = 1.0
            else:
                axis_low = q10
                axis_high = q90
            if abs(axis_high - axis_low) < 1e-9:
                for candidate in [q95, q97, q99, raw_max]:
                    if candidate > axis_low + 1e-9:
                        axis_high = candidate
                        break
            if abs(axis_high - axis_low) < 1e-9:
                for candidate in [q05, raw_min]:
                    if candidate < axis_high - 1e-9:
                        axis_low = candidate
                        break
            if abs(axis_high - axis_low) < 1e-9:
                continue
            axis_specs.append(
                AxisSpec(
                    name=f"sensitive_x_{col}",
                    feature=col,
                    column=z_col,
                    low=float((axis_low - median) / scale),
                    high=float((axis_high - median) / scale),
                    low_raw=axis_low,
                    high_raw=axis_high,
                    missing_column=miss_col,
                )
            )

    if categorical:
        cat = frame[categorical].copy()
        for col in categorical:
            cat[col] = cat[col].astype(str).str.strip().replace({"nan": "Missing", "NA": "Missing", "": "Missing"})
        dummies = pd.get_dummies(cat, prefix=categorical, dtype=np.float32)
        parts.append(dummies)
        input_columns.extend(dummies.columns.tolist())

    x = pd.concat(parts, axis=1).to_numpy(dtype=np.float32)
    prep_summary = {
        "continuous": continuous_summary,
        "n_input_columns": len(input_columns),
        "axis_specs": [asdict(axis) for axis in axis_specs],
        "label_rate": float(y.mean()),
        "sensitive_rate": float(s.mean()),
    }
    return x, input_columns, axis_specs, prep_summary


def load_adult(args: argparse.Namespace, seed: int) -> PreparedData:
    download_adult_data(args.adult_dir)
    train = pd.read_csv(args.adult_dir / "adult.data", names=ADULT_COLUMNS, header=None, na_values="?", skipinitialspace=True)
    test = pd.read_csv(args.adult_dir / "adult.test", names=ADULT_COLUMNS, header=None, na_values="?", skipinitialspace=True, comment="|")
    frame = pd.concat([train, test], ignore_index=True).dropna().reset_index(drop=True)
    if args.max_rows is not None:
        frame = frame.sample(n=min(args.max_rows, len(frame)), random_state=seed).reset_index(drop=True)
    frame["income"] = frame["income"].astype(str).str.replace(".", "", regex=False).str.strip()
    y = (frame["income"] == ">50K").to_numpy(dtype=np.float32)
    s = (frame["sex"].astype(str).str.strip() == "Female").to_numpy(dtype=np.float32)
    adult_axis_preset = str(getattr(args, "adult_axis_preset", "default"))
    engineered_proxy_features: list[str] = []
    if adult_axis_preset in {"gender_proxy", "gender_proxy_no_sparse"}:
        relationship = frame["relationship"].astype(str).str.strip()
        marital = frame["marital_status"].astype(str).str.strip()
        occupation = frame["occupation"].astype(str).str.strip()
        frame["relationship_spouse"] = relationship.isin(["Husband", "Wife"]).astype(np.float32)
        frame["relationship_own_child"] = (relationship == "Own-child").astype(np.float32)
        frame["marital_married"] = (marital == "Married-civ-spouse").astype(np.float32)
        frame["marital_never_married"] = (marital == "Never-married").astype(np.float32)
        frame["occupation_exec_prof"] = occupation.isin(["Exec-managerial", "Prof-specialty"]).astype(np.float32)
        frame["occupation_service"] = occupation.isin(["Other-service", "Priv-house-serv", "Handlers-cleaners"]).astype(np.float32)
        engineered_proxy_features = [
            "relationship_spouse",
            "relationship_own_child",
            "marital_married",
            "marital_never_married",
            "occupation_exec_prof",
            "occupation_service",
        ]
    rng = np.random.default_rng(seed)
    train_idx, val_idx, test_idx = stratified_indices(y, s, rng, args.val_frac, args.test_frac)
    continuous = ["age", "fnlwgt", "education_num", "capital_gain", "capital_loss", "hours_per_week"] + engineered_proxy_features
    if adult_axis_preset in {"gender_proxy", "gender_proxy_no_sparse"}:
        categorical = ["workclass", "race", "native_country"]
    else:
        categorical = ["workclass", "marital_status", "occupation", "relationship", "race", "native_country"]
    if adult_axis_preset == "gender_proxy":
        axis_features = ["age", "education_num", "capital_gain", "capital_loss", "hours_per_week"] + engineered_proxy_features
    elif adult_axis_preset == "gender_proxy_no_sparse":
        axis_features = ["age", "education_num", "hours_per_week"] + engineered_proxy_features
    else:
        axis_features = ["age", "education_num", "capital_gain", "capital_loss", "hours_per_week"]
    x, input_columns, axes, prep_summary = prepare_tabular_frame(
        frame=frame,
        y=y,
        s=s,
        sensitive_column="S_female",
        continuous=continuous,
        categorical=categorical,
        axis_features=axis_features,
        train_idx=train_idx,
        binary_axis_features=engineered_proxy_features,
    )
    spec = DatasetSpec(
        name="adult",
        sensitive_name="sex",
        sensitive_positive="Female",
        sensitive_negative="Male",
        label_name="income",
        favorable_label=">50K",
        input_columns=input_columns,
        sensitive_column="S_female",
        continuous_columns=continuous,
        categorical_columns=categorical,
        axes=axes,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        source_summary={
            "rows": int(len(frame)),
            "raw_source": "UCI Adult/Census Income",
            "task": "Predict income >50K.",
            "adult_axis_preset": adult_axis_preset,
            "engineered_proxy_features": engineered_proxy_features,
            "preprocessing": prep_summary,
        },
    )
    return PreparedData(name="adult", x=x, y=y, s=s, spec=spec)


def load_acs_employment(args: argparse.Namespace, seed: int) -> PreparedData:
    try:
        from folktables import ACSDataSource
    except ImportError as exc:
        raise RuntimeError(
            "The `acs_employment` dataset requires the `folktables` package. "
            "Install it with `python3 -m pip install folktables`."
        ) from exc

    states = [str(state).upper() for state in args.acs_states]
    source = ACSDataSource(
        survey_year=str(args.acs_year),
        horizon=str(args.acs_horizon),
        survey="person",
        root_dir=str(args.acs_root),
    )
    frame = source.get_data(states=states, density=float(args.acs_density), random_seed=seed, download=True).reset_index(drop=True)
    frame = frame[(pd.to_numeric(frame["AGEP"], errors="coerce") > 16) & (pd.to_numeric(frame["AGEP"], errors="coerce") < 90)]
    if "PWGTP" in frame.columns:
        frame = frame[pd.to_numeric(frame["PWGTP"], errors="coerce") >= 1]
    frame = frame.reset_index(drop=True)
    if args.max_rows is not None:
        frame = frame.sample(n=min(args.max_rows, len(frame)), random_state=seed).reset_index(drop=True)

    age = pd.to_numeric(frame["AGEP"], errors="coerce")
    y = (pd.to_numeric(frame["ESR"], errors="coerce") == 1).to_numpy(dtype=np.float32)
    s = (age >= 40).to_numpy(dtype=np.float32)

    def num(col: str) -> pd.Series:
        return pd.to_numeric(frame[col], errors="coerce")

    frame["female"] = (num("SEX") == 2).astype(np.float32)
    frame["married"] = (num("MAR") == 1).astype(np.float32)
    frame["never_married"] = (num("MAR") == 5).astype(np.float32)
    frame["divorced_or_separated"] = num("MAR").isin([3, 4]).astype(np.float32)
    frame["widowed"] = (num("MAR") == 2).astype(np.float32)
    frame["relp_householder"] = (num("RELP") == 0).astype(np.float32)
    frame["relp_spouse"] = (num("RELP") == 1).astype(np.float32)
    frame["relp_child"] = num("RELP").isin([2, 3, 4]).astype(np.float32)
    frame["relp_other"] = (~num("RELP").isin([0, 1, 2, 3, 4])).astype(np.float32)
    frame["native_us"] = (num("NATIVITY") == 1).astype(np.float32)
    frame["citizen"] = num("CIT").isin([1, 2, 3, 4]).astype(np.float32)
    frame["recent_mover"] = num("MIG").isin([2, 3]).astype(np.float32)
    frame["military_service"] = num("MIL").isin([1, 2, 3, 4]).astype(np.float32)
    frame["any_disability"] = (num("DIS") == 1).astype(np.float32)
    frame["hearing_disability"] = (num("DEAR") == 1).astype(np.float32)
    frame["vision_disability"] = (num("DEYE") == 1).astype(np.float32)
    frame["cognitive_disability"] = (num("DREM") == 1).astype(np.float32)
    frame["race_white"] = (num("RAC1P") == 1).astype(np.float32)
    frame["race_black"] = (num("RAC1P") == 2).astype(np.float32)
    frame["race_asian"] = (num("RAC1P") == 6).astype(np.float32)
    frame["race_other_or_multiracial"] = (~num("RAC1P").isin([1, 2, 6])).astype(np.float32)
    frame["ancestry_reported"] = num("ANC").isin([1, 2]).astype(np.float32)

    rng = np.random.default_rng(seed)
    train_idx, val_idx, test_idx = stratified_indices(y, s, rng, args.val_frac, args.test_frac)
    binary_features = [
        "female",
        "married",
        "never_married",
        "divorced_or_separated",
        "widowed",
        "relp_householder",
        "relp_spouse",
        "relp_child",
        "relp_other",
        "native_us",
        "citizen",
        "recent_mover",
        "military_service",
        "any_disability",
        "hearing_disability",
        "vision_disability",
        "cognitive_disability",
        "race_white",
        "race_black",
        "race_asian",
        "race_other_or_multiracial",
        "ancestry_reported",
    ]
    continuous = ["SCHL"] + binary_features
    categorical: list[str] = []
    axis_features = ["SCHL"] + binary_features
    x, input_columns, axes, prep_summary = prepare_tabular_frame(
        frame=frame,
        y=y,
        s=s,
        sensitive_column="S_age40plus",
        continuous=continuous,
        categorical=categorical,
        axis_features=axis_features,
        train_idx=train_idx,
        binary_axis_features=binary_features,
    )
    state_slug = "_".join(states).lower()
    spec = DatasetSpec(
        name=f"acs_employment_{state_slug}",
        sensitive_name="age",
        sensitive_positive="AGEP >= 40",
        sensitive_negative="AGEP < 40",
        label_name="ESR",
        favorable_label="employed",
        input_columns=input_columns,
        sensitive_column="S_age40plus",
        continuous_columns=continuous,
        categorical_columns=categorical,
        axes=axes,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        source_summary={
            "rows": int(len(frame)),
            "raw_source": "Folktables ACS Employment",
            "task": "Predict employment from ACS person records.",
            "protected_attribute": "Age binarized as protected when AGEP >= 40.",
            "states": states,
            "survey_year": str(args.acs_year),
            "horizon": str(args.acs_horizon),
            "density": float(args.acs_density),
            "filters": ["AGEP > 16", "AGEP < 90", "PWGTP >= 1 when available"],
            "dropped_columns": ["AGEP"],
            "preprocessing": prep_summary,
        },
    )
    return PreparedData(name=spec.name, x=x, y=y, s=s, spec=spec)


def load_bank(args: argparse.Namespace, seed: int) -> PreparedData:
    download_bank_data(args.bank_dir)
    frame = pd.read_csv(args.bank_dir / "bank-full.csv", sep=";")
    if args.max_rows is not None:
        frame = frame.sample(n=min(args.max_rows, len(frame)), random_state=seed).reset_index(drop=True)
    else:
        frame = frame.reset_index(drop=True)

    y = (frame["y"].astype(str).str.strip() == "yes").to_numpy(dtype=np.float32)
    age = as_float_series(frame["age"])
    s = ((age < 25) | (age >= 60)).to_numpy(dtype=np.float32)
    rng = np.random.default_rng(seed)
    train_idx, val_idx, test_idx = stratified_indices(y, s, rng, args.val_frac, args.test_frac)

    # The fairness benchmark convention binarizes age into a protected group.
    # We omit raw age and call duration: age would leak the protected attribute,
    # while duration is a post-contact variable not available before a call.
    continuous = ["balance", "day", "campaign", "pdays", "previous"]
    categorical = ["job", "marital", "education", "default", "housing", "loan", "contact", "month", "poutcome"]
    axis_features = ["balance", "campaign", "pdays", "previous"]
    x, input_columns, axes, prep_summary = prepare_tabular_frame(
        frame=frame,
        y=y,
        s=s,
        sensitive_column="S_age_unprivileged",
        continuous=continuous,
        categorical=categorical,
        axis_features=axis_features,
        train_idx=train_idx,
    )
    spec = DatasetSpec(
        name="bank",
        sensitive_name="age",
        sensitive_positive="age <25 or age >=60",
        sensitive_negative="age >=25 and age <60",
        label_name="deposit",
        favorable_label="subscribed term deposit",
        input_columns=input_columns,
        sensitive_column="S_age_unprivileged",
        continuous_columns=continuous,
        categorical_columns=categorical,
        axes=axes,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        source_summary={
            "rows": int(len(frame)),
            "raw_source": BANK_URL,
            "task": "Predict whether the client subscribed a term deposit using the full UCI bank-full.csv dataset.",
            "protected_attribute": "Age binarized as unprivileged when age <25 or age >=60.",
            "dropped_columns": ["age", "duration"],
            "preprocessing": prep_summary,
        },
    )
    return PreparedData(name="bank", x=x, y=y, s=s, spec=spec)


def load_compas(args: argparse.Namespace, seed: int) -> PreparedData:
    download_compas(args.compas_path)
    frame = pd.read_csv(args.compas_path)
    frame = frame[
        frame["race"].isin(["African-American", "Caucasian"])
        & (frame["days_b_screening_arrest"] <= 30)
        & (frame["days_b_screening_arrest"] >= -30)
        & (frame["is_recid"] != -1)
        & (frame["c_charge_degree"] != "O")
        & (frame["score_text"] != "N/A")
    ].copy()
    frame["c_jail_in_dt"] = pd.to_datetime(frame["c_jail_in"], errors="coerce")
    frame["c_jail_out_dt"] = pd.to_datetime(frame["c_jail_out"], errors="coerce")
    frame["jail_days"] = (frame["c_jail_out_dt"] - frame["c_jail_in_dt"]).dt.total_seconds() / 86400.0
    frame["jail_days"] = frame["jail_days"].clip(lower=0.0)
    frame = frame.reset_index(drop=True)
    if args.max_rows is not None:
        frame = frame.sample(n=min(args.max_rows, len(frame)), random_state=seed).reset_index(drop=True)

    y = (pd.to_numeric(frame["two_year_recid"], errors="coerce") == 0).to_numpy(dtype=np.float32)
    s = (frame["race"].astype(str).str.strip() == "African-American").to_numpy(dtype=np.float32)
    rng = np.random.default_rng(seed)
    train_idx, val_idx, test_idx = stratified_indices(y, s, rng, args.val_frac, args.test_frac)

    continuous = ["age", "priors_count", "juv_fel_count", "juv_misd_count", "juv_other_count", "jail_days"]
    categorical = ["sex", "c_charge_degree"]
    axis_features = ["age", "priors_count", "juv_fel_count", "juv_misd_count", "juv_other_count", "jail_days"]
    x, input_columns, axes, prep_summary = prepare_tabular_frame(
        frame=frame,
        y=y,
        s=s,
        sensitive_column="S_african_american",
        continuous=continuous,
        categorical=categorical,
        axis_features=axis_features,
        train_idx=train_idx,
    )
    spec = DatasetSpec(
        name="compas",
        sensitive_name="race",
        sensitive_positive="African-American",
        sensitive_negative="Caucasian",
        label_name="two_year_recid",
        favorable_label="no two-year recidivism",
        input_columns=input_columns,
        sensitive_column="S_african_american",
        continuous_columns=continuous,
        categorical_columns=categorical,
        axes=axes,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        source_summary={
            "rows": int(len(frame)),
            "raw_source": COMPAS_URL,
            "task": "Predict no two-year recidivism on the ProPublica COMPAS two-year dataset.",
            "protected_attribute": "Race, restricted to African-American and Caucasian defendants.",
            "filters": [
                "days_b_screening_arrest between -30 and 30",
                "is_recid != -1",
                "c_charge_degree != O",
                "score_text != N/A",
            ],
            "excluded_features": ["race", "decile_score", "score_text", "v_score_text"],
            "preprocessing": prep_summary,
        },
    )
    return PreparedData(name="compas", x=x, y=y, s=s, spec=spec)


def load_default_credit(args: argparse.Namespace, seed: int) -> PreparedData:
    download_default_credit(args.default_credit_path)
    frame = pd.read_excel(args.default_credit_path, header=1).reset_index(drop=True)
    if args.max_rows is not None:
        frame = frame.sample(n=min(args.max_rows, len(frame)), random_state=seed).reset_index(drop=True)

    target_col = "default payment next month"
    y = (pd.to_numeric(frame[target_col], errors="coerce") == 0).to_numpy(dtype=np.float32)
    s = (pd.to_numeric(frame["SEX"], errors="coerce") == 2).to_numpy(dtype=np.float32)
    rng = np.random.default_rng(seed)
    train_idx, val_idx, test_idx = stratified_indices(y, s, rng, args.val_frac, args.test_frac)

    continuous = [
        "LIMIT_BAL",
        "AGE",
        "PAY_0",
        "PAY_2",
        "PAY_3",
        "PAY_4",
        "PAY_5",
        "PAY_6",
        "BILL_AMT1",
        "BILL_AMT2",
        "BILL_AMT3",
        "BILL_AMT4",
        "BILL_AMT5",
        "BILL_AMT6",
        "PAY_AMT1",
        "PAY_AMT2",
        "PAY_AMT3",
        "PAY_AMT4",
        "PAY_AMT5",
        "PAY_AMT6",
    ]
    categorical = ["EDUCATION", "MARRIAGE"]
    axis_features = ["LIMIT_BAL", "AGE", "PAY_0", "BILL_AMT1", "PAY_AMT1"]
    x, input_columns, axes, prep_summary = prepare_tabular_frame(
        frame=frame,
        y=y,
        s=s,
        sensitive_column="S_female",
        continuous=continuous,
        categorical=categorical,
        axis_features=axis_features,
        train_idx=train_idx,
    )
    spec = DatasetSpec(
        name="default_credit",
        sensitive_name="sex",
        sensitive_positive="Female",
        sensitive_negative="Male",
        label_name="default payment next month",
        favorable_label="no default next month",
        input_columns=input_columns,
        sensitive_column="S_female",
        continuous_columns=continuous,
        categorical_columns=categorical,
        axes=axes,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        source_summary={
            "rows": int(len(frame)),
            "raw_source": DEFAULT_CREDIT_URL,
            "task": "Predict no default next month on the UCI Default of Credit Card Clients dataset.",
            "protected_attribute": "Sex, with Female as the sensitive-positive group.",
            "preprocessing": prep_summary,
        },
    )
    return PreparedData(name="default_credit", x=x, y=y, s=s, spec=spec)


def load_hmda(args: argparse.Namespace, seed: int) -> PreparedData:
    download_hmda(args.hmda_path, args.hmda_url)
    usecols = [
        "derived_race",
        "derived_sex",
        "action_taken",
        "loan_type",
        "loan_purpose",
        "lien_status",
        "preapproval",
        "occupancy_type",
        "loan_amount",
        "loan_to_value_ratio",
        "property_value",
        "income",
        "debt_to_income_ratio",
        "applicant_age",
        "tract_minority_population_percent",
        "tract_to_msa_income_percentage",
        "ffiec_msa_md_median_family_income",
    ]
    frame = pd.read_csv(args.hmda_path, usecols=usecols, low_memory=False)
    frame = frame[
        frame["derived_race"].isin(["White", "Black or African American"])
        & frame["action_taken"].isin([1, 2, 3])
    ].reset_index(drop=True)
    if args.max_rows is not None:
        frame = frame.sample(n=min(args.max_rows, len(frame)), random_state=seed).reset_index(drop=True)
    y = frame["action_taken"].isin([1, 2]).to_numpy(dtype=np.float32)
    s = (frame["derived_race"] == "Black or African American").to_numpy(dtype=np.float32)
    rng = np.random.default_rng(seed)
    train_idx, val_idx, test_idx = stratified_indices(y, s, rng, args.val_frac, args.test_frac)
    continuous = [
        "loan_amount",
        "loan_to_value_ratio",
        "property_value",
        "income",
        "debt_to_income_ratio",
        "tract_minority_population_percent",
        "tract_to_msa_income_percentage",
        "ffiec_msa_md_median_family_income",
    ]
    categorical = ["derived_sex", "loan_type", "loan_purpose", "lien_status", "preapproval", "occupancy_type", "applicant_age"]
    axis_features = [
        "loan_amount",
        "loan_to_value_ratio",
        "property_value",
        "income",
        "debt_to_income_ratio",
        "tract_minority_population_percent",
        "tract_to_msa_income_percentage",
    ]
    x, input_columns, axes, prep_summary = prepare_tabular_frame(
        frame=frame,
        y=y,
        s=s,
        sensitive_column="S_black",
        continuous=continuous,
        categorical=categorical,
        axis_features=axis_features,
        train_idx=train_idx,
    )
    spec = DatasetSpec(
        name="hmda",
        sensitive_name="derived_race",
        sensitive_positive="Black or African American",
        sensitive_negative="White",
        label_name="action_taken",
        favorable_label="Originated or approved but not accepted",
        input_columns=input_columns,
        sensitive_column="S_black",
        continuous_columns=continuous,
        categorical_columns=categorical,
        axes=axes,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        source_summary={
            "rows": int(len(frame)),
            "raw_rows": int(pd.read_csv(args.hmda_path, usecols=["action_taken"]).shape[0]),
            "raw_source": args.hmda_url,
            "task": "Predict favorable HMDA action_taken in {1,2} vs denial 3 for White and Black applicants.",
            "preprocessing": prep_summary,
        },
    )
    return PreparedData(name="hmda", x=x, y=y, s=s, spec=spec)


def prepared_cache_key(args: argparse.Namespace, seed: int) -> str:
    payload = {
        "version": 1,
        "dataset": args.dataset,
        "seed": int(seed),
        "val_frac": float(args.val_frac),
        "test_frac": float(args.test_frac),
        "max_rows": args.max_rows,
        "adult_axis_preset": args.adult_axis_preset,
        "acs_year": args.acs_year,
        "acs_horizon": args.acs_horizon,
        "acs_states": [str(s).upper() for s in args.acs_states],
        "acs_density": float(args.acs_density),
        "hmda_path": str(args.hmda_path),
        "hmda_url": str(args.hmda_url),
        "bank_dir": str(args.bank_dir),
        "compas_path": str(args.compas_path),
        "default_credit_path": str(args.default_credit_path),
    }
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:24]


def load_data_uncached(args: argparse.Namespace, seed: int) -> PreparedData:
    if args.dataset == "adult":
        data = load_adult(args, seed)
    elif args.dataset == "acs_employment":
        data = load_acs_employment(args, seed)
    elif args.dataset == "bank":
        data = load_bank(args, seed)
    elif args.dataset == "compas":
        data = load_compas(args, seed)
    elif args.dataset == "default_credit":
        data = load_default_credit(args, seed)
    else:
        data = load_hmda(args, seed)
    data.s_placebo = label_stratified_placebo_sensitive(data.s, data.y, seed)
    return data


def load_data(args: argparse.Namespace, seed: int) -> PreparedData:
    if args.prepared_cache_dir is None:
        return load_data_uncached(args, seed)

    cache_dir = args.prepared_cache_dir / args.dataset
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{prepared_cache_key(args, seed)}.pkl"
    if cache_path.exists():
        with cache_path.open("rb") as handle:
            return pickle.load(handle)

    data = load_data_uncached(args, seed)
    tmp_path = cache_path.with_name(f"{cache_path.name}.{os.getpid()}.tmp")
    with tmp_path.open("wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp_path, cache_path)
    return data


def tensor_slice(array: np.ndarray, indices: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(array[indices], dtype=torch.float32, device=device)


def iter_minibatches(indices: np.ndarray, batch_size: int, rng: np.random.Generator) -> Iterable[np.ndarray]:
    shuffled = indices.copy()
    rng.shuffle(shuffled)
    for start in range(0, len(shuffled), batch_size):
        yield shuffled[start : start + batch_size]


def make_model(input_dim: int, args: argparse.Namespace, device: torch.device) -> MLP:
    return MLP(input_dim=input_dim, hidden_dim=args.hidden_dim, depth=args.depth, dropout=args.dropout).to(device)


def make_mobius_fairness_model(spec: DatasetSpec, args: argparse.Namespace, device: torch.device) -> MobiusFairnessHead:
    column_to_idx = {name: i for i, name in enumerate(spec.input_columns)}
    sensitive_col = column_to_idx[spec.sensitive_column]
    axis_cols = [column_to_idx[axis.column] for axis in spec.axes]
    context_mask_cols = [sensitive_col, *axis_cols]
    for axis in spec.axes:
        if axis.missing_column is not None and axis.missing_column in column_to_idx:
            context_mask_cols.append(column_to_idx[axis.missing_column])
    axis_thresholds = [(axis.low + axis.high) / 2.0 for axis in spec.axes]
    axis_scales = [max(abs(axis.high - axis.low) / 8.0, 1e-3) for axis in spec.axes]
    return MobiusFairnessHead(
        input_dim=len(spec.input_columns),
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        dropout=args.dropout,
        sensitive_col=sensitive_col,
        axis_cols=axis_cols,
        context_mask_cols=context_mask_cols,
        axis_thresholds=axis_thresholds,
        axis_scales=axis_scales,
        toggle_mode=args.fairness_mobius_toggle_mode,
    ).to(device)


def make_adversary(args: argparse.Namespace, device: torch.device) -> Adversary:
    # The adversary is label-conditioned, so it targets residual group information
    # within outcome strata rather than only demographic parity.
    return Adversary(input_dim=args.hidden_dim + 1, hidden_dim=args.adv_hidden_dim).to(device)


def clone_with_sensitive(x: torch.Tensor, sensitive_col: int, value: float) -> torch.Tensor:
    out = x.clone()
    out[:, sensitive_col] = value
    return out


def clone_with_axis(x: torch.Tensor, axis: AxisSpec, column_to_idx: dict[str, int], value: float) -> torch.Tensor:
    out = x.clone()
    out[:, column_to_idx[axis.column]] = value
    if axis.missing_column is not None and axis.missing_column in column_to_idx:
        out[:, column_to_idx[axis.missing_column]] = 0.0
    return out


def sensitive_effect_logits(model: nn.Module, x: torch.Tensor, sensitive_col: int) -> torch.Tensor:
    x0 = clone_with_sensitive(x, sensitive_col, 0.0)
    x1 = clone_with_sensitive(x, sensitive_col, 1.0)
    return model(x1) - model(x0)


def pair_interaction_logits(model: nn.Module, x: torch.Tensor, sensitive_col: int, axis: AxisSpec, column_to_idx: dict[str, int]) -> torch.Tensor:
    x_low = clone_with_axis(x, axis, column_to_idx, axis.low)
    x_high = clone_with_axis(x, axis, column_to_idx, axis.high)
    d_low = sensitive_effect_logits(model, x_low, sensitive_col)
    d_high = sensitive_effect_logits(model, x_high, sensitive_col)
    return d_high - d_low


def axis_effect_logits(model: nn.Module, x: torch.Tensor, axis: AxisSpec, column_to_idx: dict[str, int]) -> torch.Tensor:
    x_low = clone_with_axis(x, axis, column_to_idx, axis.low)
    x_high = clone_with_axis(x, axis, column_to_idx, axis.high)
    return model(x_high) - model(x_low)


def two_axis_effect_logits(
    model: nn.Module,
    x: torch.Tensor,
    effect_axis: AxisSpec,
    context_axis: AxisSpec,
    column_to_idx: dict[str, int],
) -> torch.Tensor:
    x_context_low = clone_with_axis(x, context_axis, column_to_idx, context_axis.low)
    x_context_high = clone_with_axis(x, context_axis, column_to_idx, context_axis.high)
    effect_low = axis_effect_logits(model, x_context_low, effect_axis, column_to_idx)
    effect_high = axis_effect_logits(model, x_context_high, effect_axis, column_to_idx)
    return effect_high - effect_low


def topk_square(values: torch.Tensor, frac: float) -> torch.Tensor:
    if values.numel() == 0:
        return values.new_tensor(0.0)
    k = max(1, int(math.ceil(values.numel() * frac)))
    return torch.topk(values.square().flatten(), k=k, largest=True).values.mean()


def fairness_loss(
    model: nn.Module,
    x_anchor: torch.Tensor,
    spec: DatasetSpec,
    column_to_idx: dict[str, int],
    args: argparse.Namespace,
    model_name: str,
) -> torch.Tensor:
    sensitive_col = column_to_idx[spec.sensitive_column]
    main = sensitive_effect_logits(model, x_anchor, sensitive_col)
    loss = args.lambda_main * main.square().mean()
    if args.lambda_main > 0.0:
        loss = loss + args.lambda_tail * topk_square(main, args.tail_frac)

    if model_name == "topology_main":
        return loss

    axes = spec.axes
    if model_name == "topology_single":
        axes = spec.axes[:1]
    elif model_name in DIRECT_WRONG_MODELS:
        axes = spec.axes

    pair_losses: list[torch.Tensor] = []
    for axis in axes:
        pair = pair_interaction_logits(model, x_anchor, sensitive_col, axis, column_to_idx)
        if model_name in DIRECT_WRONG_MODELS:
            margin = pair.new_tensor(float(args.wrong_effect_margin))
            wrong_pair = torch.relu(margin - pair.abs())
            pair_losses.append(wrong_pair.square().mean() + args.lambda_tail * topk_square(wrong_pair, args.tail_frac))
        else:
            pair_losses.append(pair.square().mean() + args.lambda_tail * topk_square(pair, args.tail_frac))
    if pair_losses:
        loss = loss + args.lambda_pair * torch.stack(pair_losses).mean()
    return loss


def group_gap_square_torch(values: torch.Tensor, s: torch.Tensor, y: torch.Tensor, min_cell_size: int) -> torch.Tensor:
    return group_gap_target_square_torch(values, s, y, min_cell_size, target=0.0)


def group_gap_target_square_torch(
    values: torch.Tensor,
    s: torch.Tensor,
    y: torch.Tensor,
    min_cell_size: int,
    target: float,
) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    s_bool = s > 0.5
    y_bool = y > 0.5
    scopes = [torch.ones_like(s_bool, dtype=torch.bool), y_bool, ~y_bool]
    target_tensor = values.new_tensor(float(target))
    for scope in scopes:
        mask0 = scope & ~s_bool
        mask1 = scope & s_bool
        if int(mask0.sum().detach().cpu()) >= min_cell_size and int(mask1.sum().detach().cpu()) >= min_cell_size:
            gap = values[mask1].mean() - values[mask0].mean()
            losses.append((gap - target_tensor).square())
    if not losses:
        return values.new_tensor(0.0)
    return torch.stack(losses).mean()


def group_gap_margin_square_torch(
    values: torch.Tensor,
    s: torch.Tensor,
    y: torch.Tensor,
    min_cell_size: int,
    margin: float,
) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    s_bool = s > 0.5
    y_bool = y > 0.5
    scopes = [torch.ones_like(s_bool, dtype=torch.bool), y_bool, ~y_bool]
    margin_tensor = values.new_tensor(float(margin))
    for scope in scopes:
        mask0 = scope & ~s_bool
        mask1 = scope & s_bool
        if int(mask0.sum().detach().cpu()) >= min_cell_size and int(mask1.sum().detach().cpu()) >= min_cell_size:
            gap = values[mask1].mean() - values[mask0].mean()
            losses.append(torch.relu(margin_tensor - gap.abs()).square())
    if not losses:
        return values.new_tensor(0.0)
    return torch.stack(losses).mean()


def weighted_group_gap_square_torch(
    values: torch.Tensor,
    s: torch.Tensor,
    y: torch.Tensor,
    min_cell_size: int,
    weights: torch.Tensor,
) -> torch.Tensor:
    return weighted_group_gap_target_square_torch(values, s, y, min_cell_size, weights, target=0.0)


def weighted_group_gap_target_square_torch(
    values: torch.Tensor,
    s: torch.Tensor,
    y: torch.Tensor,
    min_cell_size: int,
    weights: torch.Tensor,
    target: float,
) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    s_bool = s > 0.5
    y_bool = y > 0.5
    scopes = [torch.ones_like(s_bool, dtype=torch.bool), y_bool, ~y_bool]
    weights = torch.clamp(weights, min=0.0)
    target_tensor = values.new_tensor(float(target))
    for scope in scopes:
        mask0 = scope & ~s_bool
        mask1 = scope & s_bool
        if int(mask0.sum().detach().cpu()) >= min_cell_size and int(mask1.sum().detach().cpu()) >= min_cell_size:
            w0 = weights[mask0]
            w1 = weights[mask1]
            if float(w0.sum().detach().cpu()) > 1e-6 and float(w1.sum().detach().cpu()) > 1e-6:
                mean0 = (values[mask0] * w0).sum() / torch.clamp(w0.sum(), min=1e-6)
                mean1 = (values[mask1] * w1).sum() / torch.clamp(w1.sum(), min=1e-6)
                losses.append((mean1 - mean0 - target_tensor).square())
    if not losses:
        return values.new_tensor(0.0)
    return torch.stack(losses).mean()


def weighted_group_gap_margin_square_torch(
    values: torch.Tensor,
    s: torch.Tensor,
    y: torch.Tensor,
    min_cell_size: int,
    weights: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    s_bool = s > 0.5
    y_bool = y > 0.5
    scopes = [torch.ones_like(s_bool, dtype=torch.bool), y_bool, ~y_bool]
    weights = torch.clamp(weights, min=0.0)
    margin_tensor = values.new_tensor(float(margin))
    for scope in scopes:
        mask0 = scope & ~s_bool
        mask1 = scope & s_bool
        if int(mask0.sum().detach().cpu()) >= min_cell_size and int(mask1.sum().detach().cpu()) >= min_cell_size:
            w0 = weights[mask0]
            w1 = weights[mask1]
            if float(w0.sum().detach().cpu()) > 1e-6 and float(w1.sum().detach().cpu()) > 1e-6:
                mean0 = (values[mask0] * w0).sum() / torch.clamp(w0.sum(), min=1e-6)
                mean1 = (values[mask1] * w1).sum() / torch.clamp(w1.sum(), min=1e-6)
                gap = mean1 - mean0
                losses.append(torch.relu(margin_tensor - gap.abs()).square())
    if not losses:
        return values.new_tensor(0.0)
    return torch.stack(losses).mean()


def boundary_weights_from_logits(logits: torch.Tensor, band: float) -> torch.Tensor:
    probs = torch.sigmoid(logits.detach())
    band = max(float(band), 1e-4)
    weights = torch.exp(-torch.abs(probs - 0.5) / band)
    return weights / torch.clamp(weights.mean(), min=1e-6)


def aggregate_loss_list(losses: list[torch.Tensor], top_frac: float) -> torch.Tensor | None:
    if not losses:
        return None
    stacked = torch.stack(losses)
    if top_frac >= 1.0 or len(losses) == 1:
        return stacked.mean()
    k = max(1, int(math.ceil(len(losses) * max(0.0, top_frac))))
    return torch.topk(stacked, k=k, largest=True).values.mean()


def slice_group_gap_loss_torch(
    values: torch.Tensor,
    x_anchor: torch.Tensor,
    s_anchor: torch.Tensor,
    y_anchor: torch.Tensor,
    spec: DatasetSpec,
    column_to_idx: dict[str, int],
    min_cell_size: int,
    top_frac: float,
    target: float = 0.0,
    amplify_margin: bool = False,
) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    for axis in spec.axes:
        axis_values = x_anchor[:, column_to_idx[axis.column]]
        for mask in [axis_values <= axis.low, axis_values >= axis.high]:
            if int(mask.sum().detach().cpu()) < 2 * min_cell_size:
                continue
            if amplify_margin:
                losses.append(group_gap_margin_square_torch(values[mask], s_anchor[mask], y_anchor[mask], min_cell_size, target))
            else:
                losses.append(group_gap_target_square_torch(values[mask], s_anchor[mask], y_anchor[mask], min_cell_size, target))
    aggregated = aggregate_loss_list(losses, top_frac)
    return values.new_tensor(0.0) if aggregated is None else aggregated


def higher_order_group_gap_loss_torch(
    model: nn.Module,
    x_anchor: torch.Tensor,
    s_anchor: torch.Tensor,
    y_anchor: torch.Tensor,
    spec: DatasetSpec,
    column_to_idx: dict[str, int],
    args: argparse.Namespace,
    boundary_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    pair_losses: list[torch.Tensor] = []
    boundary_losses: list[torch.Tensor] = []
    for effect_axis in spec.axes:
        for context_axis in spec.axes:
            if effect_axis.name == context_axis.name:
                continue
            effect = two_axis_effect_logits(model, x_anchor, effect_axis, context_axis, column_to_idx)
            pair_losses.append(
                group_gap_target_square_torch(
                    effect,
                    s_anchor,
                    y_anchor,
                    args.min_proxy_cell_size,
                    target=0.0,
                )
            )
            if boundary_weights is not None:
                boundary_losses.append(
                    weighted_group_gap_target_square_torch(
                        effect,
                        s_anchor,
                        y_anchor,
                        args.min_proxy_cell_size,
                        boundary_weights,
                        target=0.0,
                    )
                )
    pair_agg = aggregate_loss_list(pair_losses, args.higher_order_top_frac)
    pair_loss = pair_agg if pair_agg is not None else x_anchor.new_tensor(0.0)
    boundary_loss = x_anchor.new_tensor(0.0)
    if boundary_weights is not None:
        boundary_agg = aggregate_loss_list(boundary_losses, args.higher_order_top_frac)
        boundary_loss = boundary_agg if boundary_agg is not None else x_anchor.new_tensor(0.0)
    return args.lambda_higher_order_pair * pair_loss + args.lambda_higher_order_boundary_pair * boundary_loss


def proxy_fairness_loss(
    model: nn.Module,
    x_anchor: torch.Tensor,
    s_anchor: torch.Tensor,
    y_anchor: torch.Tensor,
    spec: DatasetSpec,
    column_to_idx: dict[str, int],
    args: argparse.Namespace,
    score_only: bool = False,
    include_boundary: bool = False,
    include_slice: bool = False,
    residual_selected: bool = False,
    score_scaffold_only: bool = False,
    include_higher_order: bool = False,
    effect_target: float = 0.0,
    amplify_gaps: bool = False,
) -> torch.Tensor:
    logits = model(x_anchor)
    if amplify_gaps:
        score_loss = group_gap_margin_square_torch(logits, s_anchor, y_anchor, args.min_proxy_cell_size, effect_target)
    else:
        score_loss = group_gap_square_torch(logits, s_anchor, y_anchor, args.min_proxy_cell_size)
    loss = args.lambda_proxy_score * score_loss
    if score_only:
        return loss
    pair_losses: list[torch.Tensor] = []
    boundary_pair_losses: list[torch.Tensor] = []
    slice_pair_losses: list[torch.Tensor] = []
    boundary_weights = boundary_weights_from_logits(logits, args.boundary_band) if (include_boundary or include_higher_order) else None
    if not score_scaffold_only:
        for axis in spec.axes:
            effect = axis_effect_logits(model, x_anchor, axis, column_to_idx)
            if amplify_gaps:
                pair_losses.append(group_gap_margin_square_torch(effect, s_anchor, y_anchor, args.min_proxy_cell_size, effect_target))
            else:
                pair_losses.append(group_gap_target_square_torch(effect, s_anchor, y_anchor, args.min_proxy_cell_size, effect_target))
            if boundary_weights is not None:
                if amplify_gaps:
                    boundary_pair_losses.append(
                        weighted_group_gap_margin_square_torch(
                            effect,
                            s_anchor,
                            y_anchor,
                            args.min_proxy_cell_size,
                            boundary_weights,
                            effect_target,
                        )
                    )
                else:
                    boundary_pair_losses.append(
                        weighted_group_gap_target_square_torch(
                            effect,
                            s_anchor,
                            y_anchor,
                            args.min_proxy_cell_size,
                            boundary_weights,
                            effect_target,
                        )
                    )
            if include_slice:
                slice_pair_losses.append(
                    slice_group_gap_loss_torch(
                        effect,
                        x_anchor,
                        s_anchor,
                        y_anchor,
                        spec,
                        column_to_idx,
                        args.min_proxy_cell_size,
                        args.slice_top_frac,
                        target=effect_target,
                        amplify_margin=amplify_gaps,
                    )
                )
        if residual_selected:
            proxy_pair_agg = aggregate_loss_list(pair_losses, args.residual_top_frac)
            proxy_pair = proxy_pair_agg if proxy_pair_agg is not None else logits.new_tensor(0.0)
        else:
            proxy_pair = torch.stack(pair_losses).mean() if pair_losses else logits.new_tensor(0.0)
        loss = loss + args.lambda_proxy_pair * proxy_pair
    if boundary_weights is not None:
        if amplify_gaps:
            boundary_score = weighted_group_gap_margin_square_torch(
                logits,
                s_anchor,
                y_anchor,
                args.min_proxy_cell_size,
                boundary_weights,
                effect_target,
            )
        else:
            boundary_score = weighted_group_gap_square_torch(logits, s_anchor, y_anchor, args.min_proxy_cell_size, boundary_weights)
        loss = loss + args.lambda_boundary_score * boundary_score
        if not score_scaffold_only:
            if residual_selected:
                boundary_pair_agg = aggregate_loss_list(boundary_pair_losses, args.residual_top_frac)
                boundary_pair = boundary_pair_agg if boundary_pair_agg is not None else logits.new_tensor(0.0)
            else:
                boundary_pair = torch.stack(boundary_pair_losses).mean() if boundary_pair_losses else logits.new_tensor(0.0)
            loss = loss + args.lambda_boundary_pair * boundary_pair
    if include_slice:
        slice_score = slice_group_gap_loss_torch(
            logits,
            x_anchor,
            s_anchor,
            y_anchor,
            spec,
            column_to_idx,
            args.min_proxy_cell_size,
            args.slice_top_frac,
            target=effect_target if amplify_gaps else 0.0,
            amplify_margin=amplify_gaps,
        )
        loss = loss + args.lambda_slice_score * slice_score
        if not score_scaffold_only:
            if residual_selected:
                slice_pair_agg = aggregate_loss_list(slice_pair_losses, args.residual_top_frac)
                slice_pair = slice_pair_agg if slice_pair_agg is not None else logits.new_tensor(0.0)
            else:
                slice_pair = torch.stack(slice_pair_losses).mean() if slice_pair_losses else logits.new_tensor(0.0)
            loss = loss + args.lambda_slice_pair * slice_pair
    if include_higher_order and not score_scaffold_only:
        loss = loss + higher_order_group_gap_loss_torch(
            model,
            x_anchor,
            s_anchor,
            y_anchor,
            spec,
            column_to_idx,
            args,
            boundary_weights=boundary_weights,
        )
    return loss


def finite_metric(value: float | None) -> float:
    if value is None:
        return 0.0
    if not math.isfinite(float(value)):
        return 0.0
    return float(value)


def sample_anchor_indices(data: PreparedData, rng: np.random.Generator, count: int) -> np.ndarray:
    train_idx = data.spec.train_idx
    y = data.y[train_idx].astype(np.int64)
    s = data.s[train_idx].astype(np.int64)
    labels = s * 2 + y
    per_cell = max(1, count // max(1, len(np.unique(labels))))
    out: list[int] = []
    for label in sorted(np.unique(labels).tolist()):
        pool = train_idx[labels == label]
        if len(pool) == 0:
            continue
        replace = len(pool) < per_cell
        out.extend(rng.choice(pool, size=per_cell, replace=replace).tolist())
    if len(out) < count:
        out.extend(rng.choice(train_idx, size=count - len(out), replace=len(train_idx) < count).tolist())
    rng.shuffle(out)
    return np.asarray(out[:count], dtype=np.int64)


def weighted_bce_loss(logits: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
    per = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    if weights is None:
        return per.mean()
    return (per * weights).sum() / torch.clamp(weights.sum(), min=1e-6)


def train_group_weights(data: PreparedData) -> np.ndarray:
    weights = np.ones(len(data.y), dtype=np.float32)
    train_idx = data.spec.train_idx
    labels = (data.s[train_idx].astype(np.int64) * 2 + data.y[train_idx].astype(np.int64)).astype(np.int64)
    counts = {label: int((labels == label).sum()) for label in np.unique(labels)}
    total = float(len(labels))
    for label, count in counts.items():
        group_weight = total / max(1.0, 4.0 * count)
        mask = (data.s.astype(np.int64) * 2 + data.y.astype(np.int64)) == label
        weights[mask] = group_weight
    return weights


def predict_probs(model: nn.Module, x: np.ndarray, indices: np.ndarray, device: torch.device, batch_size: int = 4096) -> np.ndarray:
    model.eval()
    probs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            idx = indices[start : start + batch_size]
            xb = torch.as_tensor(x[idx], dtype=torch.float32, device=device)
            probs.append(torch.sigmoid(model(xb)).detach().cpu().numpy())
    return np.concatenate(probs).astype(np.float32)


def threshold_predictions(prob: np.ndarray, s: np.ndarray, thresholds: dict[int, float] | None = None) -> np.ndarray:
    if thresholds is None:
        return (prob >= 0.5).astype(np.int64)
    pred = np.zeros_like(prob, dtype=np.int64)
    for group in [0, 1]:
        mask = s.astype(np.int64) == group
        pred[mask] = (prob[mask] >= thresholds.get(group, 0.5)).astype(np.int64)
    return pred


def safe_auc(y: np.ndarray, prob: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, prob))


def rate_gap(values: np.ndarray, s: np.ndarray) -> float:
    if (s == 0).sum() == 0 or (s == 1).sum() == 0:
        return float("nan")
    return float(abs(values[s == 1].mean() - values[s == 0].mean()))


def group_gap_np(values: np.ndarray, s: np.ndarray, y: np.ndarray | None = None) -> dict[str, float]:
    s_int = s.astype(np.int64)
    out: dict[str, float] = {}
    if (s_int == 0).any() and (s_int == 1).any():
        out["overall"] = float(abs(values[s_int == 1].mean() - values[s_int == 0].mean()))
    else:
        out["overall"] = float("nan")
    if y is not None:
        y_int = y.astype(np.int64)
        gaps: list[float] = []
        for label in [0, 1]:
            mask0 = (s_int == 0) & (y_int == label)
            mask1 = (s_int == 1) & (y_int == label)
            key = f"y{label}"
            if mask0.any() and mask1.any():
                gap = float(abs(values[mask1].mean() - values[mask0].mean()))
                out[key] = gap
                gaps.append(gap)
            else:
                out[key] = float("nan")
        out["conditional_mean"] = float(np.mean(gaps)) if gaps else float("nan")
        out["conditional_max"] = float(np.max(gaps)) if gaps else float("nan")
    return out


def group_metrics(y: np.ndarray, s: np.ndarray, prob: np.ndarray, thresholds: dict[int, float] | None = None) -> dict[str, float]:
    pred = threshold_predictions(prob, s, thresholds)
    out: dict[str, float] = {
        "auc": safe_auc(y, prob),
        "accuracy": float(accuracy_score(y, pred)),
        "bce": float(log_loss(y, np.clip(prob, 1e-6, 1.0 - 1e-6), labels=[0, 1])),
        "score_gap": rate_gap(prob, s),
        "dp_gap": rate_gap(pred.astype(np.float32), s),
    }
    tpr_by_group: dict[int, float] = {}
    fpr_by_group: dict[int, float] = {}
    for group in [0, 1]:
        g = s.astype(np.int64) == group
        pos = g & (y > 0.5)
        neg = g & (y <= 0.5)
        tpr_by_group[group] = float(pred[pos].mean()) if pos.any() else float("nan")
        fpr_by_group[group] = float(pred[neg].mean()) if neg.any() else float("nan")
    out["eod_gap"] = float(abs(tpr_by_group[1] - tpr_by_group[0]))
    out["fpr_gap"] = float(abs(fpr_by_group[1] - fpr_by_group[0]))
    out["aod_gap"] = 0.5 * (out["eod_gap"] + out["fpr_gap"])
    out["eodds_max_gap"] = max(out["eod_gap"], out["fpr_gap"])
    out["positive_rate_s0"] = float(pred[s == 0].mean())
    out["positive_rate_s1"] = float(pred[s == 1].mean())
    out["tpr_s0"] = tpr_by_group[0]
    out["tpr_s1"] = tpr_by_group[1]
    out["fpr_s0"] = fpr_by_group[0]
    out["fpr_s1"] = fpr_by_group[1]
    return out


def threshold_curve(y: np.ndarray, prob: np.ndarray, grid: np.ndarray) -> dict[str, np.ndarray]:
    pred = prob[None, :] >= grid[:, None]
    y_bool = y[None, :] > 0.5
    correct = (pred == y_bool).sum(axis=1).astype(np.float64)
    pos_rate = pred.mean(axis=1).astype(np.float64)
    pos = y > 0.5
    neg = ~pos
    tpr = pred[:, pos].mean(axis=1).astype(np.float64) if pos.any() else np.full(len(grid), np.nan)
    fpr = pred[:, neg].mean(axis=1).astype(np.float64) if neg.any() else np.full(len(grid), np.nan)
    return {"correct": correct, "pos_rate": pos_rate, "tpr": tpr, "fpr": fpr}


def choose_group_thresholds(y: np.ndarray, s: np.ndarray, prob: np.ndarray, grid_size: int) -> dict[int, float]:
    grid = np.linspace(0.05, 0.95, max(3, grid_size))
    mask0 = s.astype(np.int64) == 0
    mask1 = s.astype(np.int64) == 1
    if not mask0.any() or not mask1.any():
        return {0: 0.5, 1: 0.5}
    c0 = threshold_curve(y[mask0], prob[mask0], grid)
    c1 = threshold_curve(y[mask1], prob[mask1], grid)
    eod = np.abs(c1["tpr"][None, :] - c0["tpr"][:, None])
    fpr = np.abs(c1["fpr"][None, :] - c0["fpr"][:, None])
    aod = 0.5 * (eod + fpr)
    correct = c0["correct"][:, None] + c1["correct"][None, :]
    accuracy = correct / float(len(y))
    score = aod + 0.02 * (1.0 - accuracy)
    best = np.unravel_index(np.nanargmin(score), score.shape)
    return {0: float(grid[best[0]]), 1: float(grid[best[1]])}


def mechanism_metrics(
    model: nn.Module,
    data: PreparedData,
    indices: np.ndarray,
    device: torch.device,
    max_eval: int = 8192,
    args: argparse.Namespace | None = None,
) -> dict[str, float]:
    spec = data.spec
    column_to_idx = {name: i for i, name in enumerate(spec.input_columns)}
    sensitive_col = column_to_idx[spec.sensitive_column]
    rng = np.random.default_rng(12345 + len(indices))
    eval_idx = indices
    if len(eval_idx) > max_eval:
        eval_idx = rng.choice(eval_idx, size=max_eval, replace=False)
    x = torch.as_tensor(data.x[eval_idx], dtype=torch.float32, device=device)
    s_eval = data.s[eval_idx].astype(np.float32)
    y_eval = data.y[eval_idx].astype(np.float32)
    model.eval()
    with torch.no_grad():
        logits = model(x)
        d = sensitive_effect_logits(model, x, sensitive_col)
        x0 = clone_with_sensitive(x, sensitive_col, 0.0)
        x1 = clone_with_sensitive(x, sensitive_col, 1.0)
        p0 = torch.sigmoid(model(x0))
        p1 = torch.sigmoid(model(x1))
        prob_diff = p1 - p0
        flip = ((p0 >= 0.5) != (p1 >= 0.5)).to(torch.float32)
        pair_values: list[torch.Tensor] = []
        proxy_effect_gaps: list[float] = []
        axis_out: dict[str, float] = {}
        effect_by_feature: dict[str, np.ndarray] = {}
        for axis in spec.axes:
            pair = pair_interaction_logits(model, x, sensitive_col, axis, column_to_idx)
            pair_abs = pair.abs().detach().cpu().numpy()
            axis_out[f"pair_abs_mean__{axis.feature}"] = float(pair_abs.mean())
            axis_out[f"pair_abs_q95__{axis.feature}"] = float(np.quantile(pair_abs, 0.95))
            pair_values.append(pair.abs())
            effect = axis_effect_logits(model, x, axis, column_to_idx).detach().cpu().numpy()
            effect_by_feature[axis.feature] = effect
            effect_gap = group_gap_np(effect, s_eval, y_eval)
            axis_out[f"proxy_effect_gap_cond__{axis.feature}"] = float(effect_gap["conditional_mean"])
            axis_out[f"proxy_effect_gap_max__{axis.feature}"] = float(effect_gap["conditional_max"])
            proxy_effect_gaps.append(float(effect_gap["conditional_mean"]))
        d_abs = d.abs().detach().cpu().numpy()
        prob_abs = prob_diff.abs().detach().cpu().numpy()
        logits_np = logits.detach().cpu().numpy()
        score_gap = group_gap_np(logits_np, s_eval, y_eval)
        probs_np = sigmoid_np(logits_np)
        x_eval_np = data.x[eval_idx]
        boundary_margin = float(getattr(args, "boundary_eval_margin", 0.15)) if args is not None else 0.15
        boundary_mask = np.abs(probs_np - 0.5) <= boundary_margin
        boundary_score_gap = group_gap_np(logits_np[boundary_mask], s_eval[boundary_mask], y_eval[boundary_mask]) if boundary_mask.any() else {}
        boundary_effect_gaps: list[float] = []
        for feature, effect in effect_by_feature.items():
            if boundary_mask.any():
                boundary_gap = group_gap_np(effect[boundary_mask], s_eval[boundary_mask], y_eval[boundary_mask])
                axis_out[f"boundary_proxy_effect_gap_cond__{feature}"] = float(boundary_gap.get("conditional_mean", float("nan")))
                if math.isfinite(float(boundary_gap.get("conditional_mean", float("nan")))):
                    boundary_effect_gaps.append(float(boundary_gap["conditional_mean"]))
            else:
                axis_out[f"boundary_proxy_effect_gap_cond__{feature}"] = float("nan")

        slice_score_gaps: list[float] = []
        slice_effect_gaps: list[float] = []
        for slice_axis in spec.axes:
            axis_values = x_eval_np[:, column_to_idx[slice_axis.column]]
            for side, slice_mask in [("low", axis_values <= slice_axis.low), ("high", axis_values >= slice_axis.high)]:
                if not slice_mask.any():
                    continue
                slice_score = group_gap_np(logits_np[slice_mask], s_eval[slice_mask], y_eval[slice_mask])
                slice_score_value = float(slice_score.get("conditional_mean", float("nan")))
                axis_out[f"slice_proxy_score_gap_cond__{slice_axis.feature}_{side}"] = slice_score_value
                if math.isfinite(slice_score_value):
                    slice_score_gaps.append(slice_score_value)
                for effect_feature, effect in effect_by_feature.items():
                    slice_effect = group_gap_np(effect[slice_mask], s_eval[slice_mask], y_eval[slice_mask])
                    slice_effect_value = float(slice_effect.get("conditional_mean", float("nan")))
                    if math.isfinite(slice_effect_value):
                        slice_effect_gaps.append(slice_effect_value)
        if pair_values:
            pair_cat = torch.cat(pair_values).detach().cpu().numpy()
        else:
            pair_cat = np.zeros(1, dtype=np.float32)
        proxy_effect_arr = np.asarray(proxy_effect_gaps if proxy_effect_gaps else [0.0], dtype=np.float32)
        boundary_effect_arr = np.asarray(boundary_effect_gaps, dtype=np.float32)
        slice_score_arr = np.asarray(slice_score_gaps, dtype=np.float32)
        slice_effect_arr = np.asarray(slice_effect_gaps, dtype=np.float32)
    out = {
        "main_logit_mean": float(d.detach().cpu().numpy().mean()),
        "main_abs_logit_mean": float(d_abs.mean()),
        "main_abs_logit_q95": float(np.quantile(d_abs, 0.95)),
        "main_abs_logit_top1pct": float(np.sort(d_abs)[-max(1, len(d_abs) // 100) :].mean()),
        "cf_abs_prob_mean": float(prob_abs.mean()),
        "cf_abs_prob_q95": float(np.quantile(prob_abs, 0.95)),
        "cf_flip_rate": float(flip.detach().cpu().numpy().mean()),
        "pair_abs_mean": float(pair_cat.mean()),
        "pair_abs_q95": float(np.quantile(pair_cat, 0.95)),
        "pair_abs_max": float(pair_cat.max()),
        "proxy_score_gap_overall": float(score_gap["overall"]),
        "proxy_score_gap_y0": float(score_gap["y0"]),
        "proxy_score_gap_y1": float(score_gap["y1"]),
        "proxy_score_gap_cond_mean": float(score_gap["conditional_mean"]),
        "proxy_score_gap_cond_max": float(score_gap["conditional_max"]),
        "proxy_effect_gap_mean": float(np.nanmean(proxy_effect_arr)),
        "proxy_effect_gap_max": float(np.nanmax(proxy_effect_arr)),
        "boundary_fraction": float(boundary_mask.mean()),
        "boundary_proxy_score_gap_cond_mean": float(boundary_score_gap.get("conditional_mean", float("nan"))),
        "boundary_proxy_score_gap_cond_max": float(boundary_score_gap.get("conditional_max", float("nan"))),
        "boundary_proxy_effect_gap_mean": float(np.nanmean(boundary_effect_arr)) if boundary_effect_arr.size else float("nan"),
        "boundary_proxy_effect_gap_max": float(np.nanmax(boundary_effect_arr)) if boundary_effect_arr.size else float("nan"),
        "slice_proxy_score_gap_mean": float(np.nanmean(slice_score_arr)) if slice_score_arr.size else float("nan"),
        "slice_proxy_score_gap_max": float(np.nanmax(slice_score_arr)) if slice_score_arr.size else float("nan"),
        "slice_proxy_effect_gap_mean": float(np.nanmean(slice_effect_arr)) if slice_effect_arr.size else float("nan"),
        "slice_proxy_effect_gap_max": float(np.nanmax(slice_effect_arr)) if slice_effect_arr.size else float("nan"),
    }
    out.update(axis_out)
    return out


def higher_order_proxy_metrics(
    model: nn.Module,
    data: PreparedData,
    indices: np.ndarray,
    device: torch.device,
    max_eval: int = 8192,
    args: argparse.Namespace | None = None,
) -> dict[str, float]:
    spec = data.spec
    if len(spec.axes) < 2:
        return {
            "higher_order_pair_count": 0.0,
            "higher_order_proxy_gap_scope_mean": float("nan"),
            "higher_order_proxy_gap_scope_max": float("nan"),
            "higher_order_boundary_proxy_gap_scope_mean": float("nan"),
            "higher_order_boundary_proxy_gap_scope_max": float("nan"),
        }
    column_to_idx = {name: i for i, name in enumerate(spec.input_columns)}
    rng = np.random.default_rng(24680 + len(indices))
    eval_idx = indices
    if len(eval_idx) > max_eval:
        eval_idx = rng.choice(eval_idx, size=max_eval, replace=False)
    x = torch.as_tensor(data.x[eval_idx], dtype=torch.float32, device=device)
    s_eval = data.s[eval_idx].astype(np.float32)
    y_eval = data.y[eval_idx].astype(np.float32)
    model.eval()
    out: dict[str, float] = {}
    scope_gaps: list[float] = []
    boundary_scope_gaps: list[float] = []
    pair_count = 0
    with torch.no_grad():
        logits = model(x)
        probs = sigmoid_np(logits.detach().cpu().numpy())
        boundary_margin = float(getattr(args, "boundary_eval_margin", 0.15)) if args is not None else 0.15
        boundary_mask = np.abs(probs - 0.5) <= boundary_margin
        for effect_axis in spec.axes:
            for context_axis in spec.axes:
                if effect_axis.name == context_axis.name:
                    continue
                pair_count += 1
                effect = two_axis_effect_logits(model, x, effect_axis, context_axis, column_to_idx).detach().cpu().numpy()
                gaps = group_gap_np(effect, s_eval, y_eval)
                key = f"{effect_axis.feature}__by__{context_axis.feature}"
                for scope in ["overall", "y0", "y1"]:
                    value = float(gaps.get(scope, float("nan")))
                    out[f"higher_order_proxy_gap_{scope}__{key}"] = value
                    if math.isfinite(value):
                        scope_gaps.append(value)
                if boundary_mask.any():
                    boundary_gaps = group_gap_np(effect[boundary_mask], s_eval[boundary_mask], y_eval[boundary_mask])
                    for scope in ["overall", "y0", "y1"]:
                        value = float(boundary_gaps.get(scope, float("nan")))
                        out[f"higher_order_boundary_proxy_gap_{scope}__{key}"] = value
                        if math.isfinite(value):
                            boundary_scope_gaps.append(value)
                else:
                    for scope in ["overall", "y0", "y1"]:
                        out[f"higher_order_boundary_proxy_gap_{scope}__{key}"] = float("nan")
    scope_arr = np.asarray(scope_gaps, dtype=np.float32)
    boundary_arr = np.asarray(boundary_scope_gaps, dtype=np.float32)
    out.update(
        {
            "higher_order_pair_count": float(pair_count),
            "higher_order_scope_count": float(len(scope_gaps)),
            "higher_order_proxy_gap_scope_mean": float(np.nanmean(scope_arr)) if scope_arr.size else float("nan"),
            "higher_order_proxy_gap_scope_max": float(np.nanmax(scope_arr)) if scope_arr.size else float("nan"),
            "higher_order_boundary_fraction": float(boundary_mask.mean()),
            "higher_order_boundary_scope_count": float(len(boundary_scope_gaps)),
            "higher_order_boundary_proxy_gap_scope_mean": float(np.nanmean(boundary_arr)) if boundary_arr.size else float("nan"),
            "higher_order_boundary_proxy_gap_scope_max": float(np.nanmax(boundary_arr)) if boundary_arr.size else float("nan"),
        }
    )
    return out


def behavior_selection_score(val_behavior: dict[str, float], args: argparse.Namespace) -> float:
    if val_behavior["auc"] < args.selector_min_auc:
        return 1e3 + (args.selector_min_auc - val_behavior["auc"])
    if val_behavior["accuracy"] < args.selector_min_acc:
        return 1e3 + (args.selector_min_acc - val_behavior["accuracy"])
    return (
        val_behavior["aod_gap"]
        + args.selector_eod_weight * val_behavior["eod_gap"]
        + args.selector_dp_weight * val_behavior["dp_gap"]
        + args.selector_auc_weight * (1.0 - val_behavior["auc"])
        + args.selector_acc_weight * (1.0 - val_behavior["accuracy"])
    )


def utility_selection_score(val_behavior: dict[str, float], args: argparse.Namespace) -> float:
    if val_behavior["auc"] < args.selector_min_auc:
        return 1e3 + (args.selector_min_auc - val_behavior["auc"])
    if val_behavior["accuracy"] < args.selector_min_acc:
        return 1e3 + (args.selector_min_acc - val_behavior["accuracy"])
    if args.utility_selector_metric == "auc":
        score = 1.0 - val_behavior["auc"]
    elif args.utility_selector_metric == "accuracy":
        score = 1.0 - val_behavior["accuracy"]
    else:
        score = val_behavior["bce"]
    if args.utility_selector_metric != "auc":
        score += args.selector_auc_weight * (1.0 - val_behavior["auc"])
    if args.utility_selector_metric != "accuracy":
        score += args.selector_acc_weight * (1.0 - val_behavior["accuracy"])
    return float(score)


def mechanism_guard_selection_score(val_behavior: dict[str, float], mechanism_residual: float, args: argparse.Namespace) -> float:
    utility = utility_selection_score(val_behavior, args)
    if utility >= 1e3:
        return utility + finite_metric(mechanism_residual)
    return finite_metric(mechanism_residual) + 1e-3 * utility


def score_control_residual(val_mechanism: dict[str, float], model_name: str, args: argparse.Namespace) -> float:
    residual = finite_metric(val_mechanism.get("proxy_score_gap_cond_mean"))
    if model_name in SPEC_BOUNDARY_MODELS:
        residual += args.selector_boundary_weight * finite_metric(val_mechanism.get("boundary_proxy_score_gap_cond_mean"))
    if model_name in SPEC_SLICE_MODELS:
        residual += args.selector_slice_weight * finite_metric(val_mechanism.get("slice_proxy_score_gap_max"))
    return residual


def wrong_spec_residual(val_mechanism: dict[str, float], model_name: str, args: argparse.Namespace) -> float:
    margin = float(args.wrong_effect_margin)
    residual = 0.0
    if model_name in DIRECT_WRONG_MODELS:
        residual += max(0.0, margin - finite_metric(val_mechanism.get("pair_abs_mean")))
    if model_name in SPEC_WRONG_EFFECT_MODELS:
        residual += max(0.0, margin - finite_metric(val_mechanism.get("proxy_score_gap_cond_mean")))
        residual += max(0.0, margin - finite_metric(val_mechanism.get("proxy_effect_gap_mean")))
        if model_name in SPEC_BOUNDARY_MODELS:
            residual += args.selector_boundary_weight * max(
                0.0,
                margin - finite_metric(val_mechanism.get("boundary_proxy_effect_gap_mean")),
            )
        if model_name in SPEC_SLICE_MODELS:
            residual += args.selector_slice_weight * max(
                0.0,
                margin - finite_metric(val_mechanism.get("slice_proxy_effect_gap_max")),
            )
    return residual


def effective_selector_policy(model_name: str, args: argparse.Namespace) -> str:
    if args.baseline_selection == "utility" and model_name in SCORE_CONTROL_SELECTION_MODELS:
        return "utility_guard_plus_score_control"
    if args.baseline_selection == "utility" and model_name in WRONG_CONTROL_SELECTION_MODELS:
        return "utility_guard_plus_wrong_spec"
    if args.selector_mode == "behavior_only" or model_name in BEHAVIOR_ONLY_SELECTION_MODELS:
        return "utility_only" if args.baseline_selection == "utility" else "endpoint_fairness_only"
    if args.selector_mode in {"behavior_direct", "behavior_proxy"} and model_name.startswith("topology"):
        if args.repair_selection == "utility_mechanism":
            return "utility_guard_plus_direct_mechanism"
        return "endpoint_fairness_plus_direct_mechanism"
    if args.selector_mode == "behavior_proxy" and (model_name in PROXY_STEERING_MODELS or model_name in SPEC_LADDER_MODELS or model_name in ADVERSARIAL_MODELS):
        if args.repair_selection == "utility_mechanism":
            return "utility_guard_plus_mechanism"
        return "endpoint_fairness_plus_mechanism"
    if model_name.startswith("topology") or model_name.startswith("proxy") or model_name.startswith("spec") or model_name.startswith("adv"):
        return "task_plus_mechanism"
    return "task_loss"


def selection_score(model_name: str, val_behavior: dict[str, float], val_mechanism: dict[str, float], args: argparse.Namespace) -> float:
    proxy_residual = val_mechanism["proxy_score_gap_cond_mean"] + val_mechanism["proxy_effect_gap_mean"]
    boundary_residual = finite_metric(val_mechanism.get("boundary_proxy_score_gap_cond_mean")) + finite_metric(
        val_mechanism.get("boundary_proxy_effect_gap_mean")
    )
    slice_residual = finite_metric(val_mechanism.get("slice_proxy_score_gap_max")) + finite_metric(
        val_mechanism.get("slice_proxy_effect_gap_max")
    )
    if args.baseline_selection == "utility" and model_name in SCORE_CONTROL_SELECTION_MODELS:
        return mechanism_guard_selection_score(val_behavior, score_control_residual(val_mechanism, model_name, args), args)
    if args.baseline_selection == "utility" and model_name in WRONG_CONTROL_SELECTION_MODELS:
        return mechanism_guard_selection_score(val_behavior, wrong_spec_residual(val_mechanism, model_name, args), args)
    if args.selector_mode == "behavior_only" or model_name in BEHAVIOR_ONLY_SELECTION_MODELS:
        if args.baseline_selection == "utility":
            return utility_selection_score(val_behavior, args)
        return behavior_selection_score(val_behavior, args)
    if args.selector_mode in {"behavior_direct", "behavior_proxy"} and model_name.startswith("topology"):
        direct_residual = val_mechanism["main_abs_logit_mean"] + val_mechanism["pair_abs_mean"]
        if args.repair_selection == "utility_mechanism":
            return mechanism_guard_selection_score(val_behavior, direct_residual, args)
        base_score = behavior_selection_score(val_behavior, args)
        return base_score + args.selector_fairness_weight * direct_residual
    if args.selector_mode == "behavior_proxy" and (model_name in PROXY_STEERING_MODELS or model_name in SPEC_LADDER_MODELS or model_name in ADVERSARIAL_MODELS):
        spec_residual = proxy_residual
        if model_name in DIRECT_TOPOLOGY_MODELS:
            spec_residual += val_mechanism["main_abs_logit_mean"] + val_mechanism["pair_abs_mean"]
        if model_name in SPEC_BOUNDARY_MODELS:
            spec_residual += args.selector_boundary_weight * boundary_residual
        if model_name in SPEC_SLICE_MODELS:
            spec_residual += args.selector_slice_weight * slice_residual
        if model_name in SPEC_HIGHER_ORDER_MODELS:
            higher_residual = finite_metric(val_mechanism.get("higher_order_proxy_gap_scope_mean")) + finite_metric(
                val_mechanism.get("higher_order_boundary_proxy_gap_scope_mean")
            )
            spec_residual += args.selector_higher_order_weight * higher_residual
        if args.repair_selection == "utility_mechanism":
            return mechanism_guard_selection_score(val_behavior, spec_residual, args)
        base_score = behavior_selection_score(val_behavior, args)
        return (
            base_score
            + args.selector_fairness_weight * spec_residual
        )
    base = val_behavior["bce"]
    if model_name.startswith("topology"):
        fair = val_mechanism["main_abs_logit_mean"] + val_mechanism["pair_abs_mean"]
        return base + args.selector_fairness_weight * fair
    if model_name.startswith("proxy") or model_name.startswith("spec"):
        fair = (
            val_mechanism["main_abs_logit_mean"]
            + val_mechanism["pair_abs_mean"]
            + proxy_residual
            + args.selector_boundary_weight * boundary_residual
            + args.selector_slice_weight * slice_residual
        )
        return base + args.selector_fairness_weight * fair
    if model_name.startswith("adv"):
        return base + args.selector_fairness_weight * proxy_residual
    return base


def train_one_model(
    model_name: str,
    data: PreparedData,
    seed: int,
    device: torch.device,
    args: argparse.Namespace,
    artifact_dir: Path | None = None,
) -> dict[str, object]:
    set_seed(seed)
    rng = np.random.default_rng(seed + 991)
    if model_name in FAIRNESS_MOBIUS_MODELS:
        model = make_mobius_fairness_model(data.spec, args, device)
    else:
        model = make_model(data.x.shape[1], args, device)
    adversary = make_adversary(args, device) if model_name in ADVERSARIAL_MODELS else None
    opt_params = list(model.parameters())
    if adversary is not None:
        opt_params.extend(adversary.parameters())
    opt = torch.optim.AdamW(opt_params, lr=args.lr, weight_decay=args.weight_decay)
    weights = train_group_weights(data) if model_name in REWEIGHTED_MODELS else np.ones(len(data.y), dtype=np.float32)
    if model_name in SENSITIVE_REMOVED_MODELS:
        sensitive_col = data.spec.input_columns.index(data.spec.sensitive_column)
        with torch.no_grad():
            first_layer = model.net[0]
            if isinstance(first_layer, nn.Linear):
                first_layer.weight[:, sensitive_col].zero_()
    else:
        sensitive_col = None
    best_state: dict[str, torch.Tensor] | None = None
    best_metrics: dict[str, object] | None = None
    best_score = float("inf")
    history: list[dict[str, object]] = []
    stale = 0
    start_time = time.time()
    epochs = min(args.epochs, 12) if args.quick else args.epochs
    print(f"[seed {seed}] training {model_name} for up to {epochs} epochs on {data.name}", flush=True)
    for epoch in range(epochs):
        model.train()
        losses: list[float] = []
        for batch_idx in iter_minibatches(data.spec.train_idx, args.batch_size, rng):
            xb = tensor_slice(data.x, batch_idx, device)
            yb = tensor_slice(data.y, batch_idx, device)
            sb = tensor_slice(data.s, batch_idx, device)
            wb = tensor_slice(weights, batch_idx, device)
            if sensitive_col is not None:
                xb = xb.clone()
                xb[:, sensitive_col] = 0.0
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = weighted_bce_loss(logits, yb, wb)
            if adversary is not None:
                warmup = min(1.0, float(epoch + 1) / max(1, args.warmup_epochs))
                features = model.features(xb)
                adv_features = gradient_reverse(features, warmup * args.lambda_adv)
                adv_input = torch.cat([adv_features, yb.unsqueeze(1)], dim=1)
                adv_logits = adversary(adv_input)
                loss = loss + nn.functional.binary_cross_entropy_with_logits(adv_logits, sb)
            if model_name in INTERACTION_STEERING_MODELS:
                anchor_idx = sample_anchor_indices(data, rng, args.anchor_batch_size)
                xa = tensor_slice(data.x, anchor_idx, device)
                sensitive_for_loss = data.s_placebo if model_name in PLACEBO_SPEC_MODELS else data.s
                assert sensitive_for_loss is not None
                sa = tensor_slice(sensitive_for_loss, anchor_idx, device)
                ya = tensor_slice(data.y, anchor_idx, device)
                if sensitive_col is not None:
                    xa = xa.clone()
                    xa[:, sensitive_col] = 0.0
                warmup = min(1.0, float(epoch + 1) / max(1, args.warmup_epochs))
                column_to_idx = {name: i for i, name in enumerate(data.spec.input_columns)}
                fair_loss = xa.new_tensor(0.0)
                if model_name in DIRECT_TOPOLOGY_MODELS:
                    fair_loss = fair_loss + fairness_loss(model, xa, data.spec, column_to_idx, args, model_name)
                if model_name in FAIRNESS_MOBIUS_MODELS:
                    fair_loss = fair_loss + fairness_loss(model, xa, data.spec, column_to_idx, args, "topology_exhaustive")
                if model_name in PROXY_STEERING_MODELS:
                    fair_loss = fair_loss + proxy_fairness_loss(
                        model,
                        xa,
                        sa,
                        ya,
                        data.spec,
                        column_to_idx,
                        args,
                        score_only=model_name in PROXY_SCORE_ONLY_MODELS,
                    )
                if model_name in SPEC_LADDER_MODELS:
                    fair_loss = fair_loss + proxy_fairness_loss(
                        model,
                        xa,
                        sa,
                        ya,
                        data.spec,
                        column_to_idx,
                        args,
                        score_only=model_name in PROXY_SCORE_ONLY_MODELS,
                        include_boundary=model_name in SPEC_BOUNDARY_MODELS,
                        include_slice=model_name in SPEC_SLICE_MODELS,
                        residual_selected=model_name in SPEC_RESIDUAL_MODELS,
                        score_scaffold_only=model_name in SPEC_FULL_SCORE_ONLY_MODELS or model_name in NO_INTERACTION_MODELS,
                        include_higher_order=model_name in SPEC_HIGHER_ORDER_MODELS,
                        effect_target=args.wrong_effect_margin if model_name in SPEC_WRONG_EFFECT_MODELS else 0.0,
                        amplify_gaps=model_name in SPEC_WRONG_EFFECT_MODELS,
                    )
                loss = loss + warmup * fair_loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            if sensitive_col is not None:
                with torch.no_grad():
                    first_layer = model.net[0]
                    if isinstance(first_layer, nn.Linear):
                        first_layer.weight[:, sensitive_col].zero_()
            losses.append(float(loss.detach().cpu()))

        val_prob = predict_probs(model, data.x, data.spec.val_idx, device)
        val_behavior = group_metrics(data.y[data.spec.val_idx], data.s[data.spec.val_idx], val_prob)
        val_mechanism = mechanism_metrics(model, data, data.spec.val_idx, device, max_eval=4096, args=args)
        if model_name in SPEC_HIGHER_ORDER_MODELS:
            val_mechanism.update(higher_order_proxy_metrics(model, data, data.spec.val_idx, device, max_eval=4096, args=args))
        score = selection_score(model_name, val_behavior, val_mechanism, args)
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": float(np.mean(losses)),
                "val_behavior": val_behavior,
                "val_mechanism": val_mechanism,
                "selection_score": float(score),
            }
        )
        if score < best_score - float(args.selection_min_delta):
            best_score = score
            stale = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_metrics = {
                "epoch": epoch + 1,
                "train_loss": float(np.mean(losses)),
                "val_behavior": val_behavior,
                "val_mechanism": val_mechanism,
                "selection_score": float(score),
            }
        else:
            stale += 1
            if stale >= args.patience:
                print(f"[seed {seed}] {model_name} early stop at epoch {epoch + 1}", flush=True)
                break
        print(
            f"[seed {seed}] {model_name} epoch {epoch + 1}: "
            f"bce={val_behavior['bce']:.4f} auc={val_behavior['auc']:.4f} "
            f"aod={val_behavior['aod_gap']:.4f} main={val_mechanism['main_abs_logit_mean']:.4f} "
            f"pair={val_mechanism['pair_abs_mean']:.4f} proxy={val_mechanism['proxy_score_gap_cond_mean']:.4f}/"
            f"{val_mechanism['proxy_effect_gap_mean']:.4f}",
            flush=True,
        )

    if best_state is not None:
        model.load_state_dict(best_state)
    assert best_metrics is not None
    val_prob = predict_probs(model, data.x, data.spec.val_idx, device)
    thresholds = choose_group_thresholds(data.y[data.spec.val_idx], data.s[data.spec.val_idx], val_prob, args.threshold_grid_size)
    test_prob = predict_probs(model, data.x, data.spec.test_idx, device)
    test_behavior = group_metrics(data.y[data.spec.test_idx], data.s[data.spec.test_idx], test_prob)
    test_behavior_post = group_metrics(data.y[data.spec.test_idx], data.s[data.spec.test_idx], test_prob, thresholds)
    test_mechanism = mechanism_metrics(model, data, data.spec.test_idx, device, args=args)
    test_mechanism.update(higher_order_proxy_metrics(model, data, data.spec.test_idx, device, args=args))
    artifact_paths: dict[str, str] = {}
    if args.save_artifacts and artifact_dir is not None:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = artifact_dir / f"{model_name}.pt"
        manifest_path = artifact_dir / f"{model_name}.artifact.json"
        checkpoint = {
            "artifact_kind": "interfair.selected_model",
            "model": model_name,
            "label": MODEL_LABELS[model_name],
            "seed": seed,
            "dataset": data.name,
            "input_dim": int(data.x.shape[1]),
            "architecture": {
                "class": model.__class__.__name__,
                "hidden_dim": int(args.hidden_dim),
                "depth": int(args.depth),
                "dropout": float(args.dropout),
                "fairness_mobius_toggle_mode": (
                    args.fairness_mobius_toggle_mode if model_name in FAIRNESS_MOBIUS_MODELS else None
                ),
                "mobius_terms": (
                    int(getattr(model, "term_masks").shape[0]) if isinstance(model, MobiusFairnessHead) else None
                ),
            },
            "input_columns": list(data.spec.input_columns),
            "sensitive_column": data.spec.sensitive_column,
            "effective_selector": effective_selector_policy(model_name, args),
            "best": to_jsonable(best_metrics),
            "group_thresholds_from_val": to_jsonable(thresholds),
            "args": artifact_args_payload(args),
            "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "adversary_state_dict": (
                {k: v.detach().cpu() for k, v in adversary.state_dict().items()} if adversary is not None else None
            ),
        }
        torch.save(checkpoint, checkpoint_path)
        manifest = {
            key: value
            for key, value in checkpoint.items()
            if key not in {"model_state_dict", "adversary_state_dict"}
        }
        manifest["checkpoint_path"] = str(checkpoint_path)
        write_json(manifest_path, manifest)
        artifact_paths = {
            "checkpoint": str(checkpoint_path),
            "manifest": str(manifest_path),
        }
    result = {
        "model": model_name,
        "label": MODEL_LABELS[model_name],
        "seed": seed,
        "effective_selector": effective_selector_policy(model_name, args),
        "best": best_metrics,
        "history": history,
        "test_behavior": test_behavior,
        "test_behavior_group_thresholds": test_behavior_post,
        "group_thresholds_from_val": thresholds,
        "test_mechanism": test_mechanism,
        "artifacts": artifact_paths,
        "train_seconds": float(time.time() - start_time),
    }
    print(
        f"[seed {seed}] finished {model_name}: "
        f"test_auc={test_behavior['auc']:.4f} test_aod={test_behavior['aod_gap']:.4f} "
        f"main={test_mechanism['main_abs_logit_mean']:.4f} pair={test_mechanism['pair_abs_mean']:.4f} "
        f"proxy={test_mechanism['proxy_score_gap_cond_mean']:.4f}/{test_mechanism['proxy_effect_gap_mean']:.4f}",
        flush=True,
    )
    return result


def dataset_summary_payload(data: PreparedData) -> dict[str, object]:
    spec = data.spec

    def split_counts(indices: np.ndarray) -> dict[str, object]:
        y = data.y[indices].astype(np.int64)
        s = data.s[indices].astype(np.int64)
        return {
            "n": int(len(indices)),
            "label_rate": float(y.mean()),
            "sensitive_rate": float(s.mean()),
            "cells": {
                f"s{group}_y{label}": int(((s == group) & (y == label)).sum())
                for group in [0, 1]
                for label in [0, 1]
            },
        }

    return {
        "dataset": data.name,
        "sensitive_name": spec.sensitive_name,
        "sensitive_positive": spec.sensitive_positive,
        "sensitive_negative": spec.sensitive_negative,
        "label_name": spec.label_name,
        "favorable_label": spec.favorable_label,
        "n_input_columns": len(spec.input_columns),
        "continuous_columns": spec.continuous_columns,
        "categorical_columns": spec.categorical_columns,
        "axes": [asdict(axis) for axis in spec.axes],
        "source_summary": spec.source_summary,
        "splits": {
            "train": split_counts(spec.train_idx),
            "val": split_counts(spec.val_idx),
            "test": split_counts(spec.test_idx),
        },
    }


def fmt(value: float) -> str:
    if value != value:
        return "nan"
    return f"{value:.4f}"


def write_report(path: Path, data: PreparedData, results: list[dict[str, object]], args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append(f"# Fairness Interaction Debugging Report: {data.name}")
    lines.append("")
    lines.append("## Data")
    lines.append("")
    lines.append(f"- Rows after task filtering: {data.spec.source_summary['rows']}")
    lines.append(f"- Sensitive toggle: `{data.spec.sensitive_name}` = `{data.spec.sensitive_positive}` vs `{data.spec.sensitive_negative}`")
    lines.append(f"- Favorable label: `{data.spec.favorable_label}`")
    lines.append(f"- Train/val/test: {len(data.spec.train_idx)} / {len(data.spec.val_idx)} / {len(data.spec.test_idx)}")
    lines.append(f"- Interaction axes: {', '.join(axis.feature for axis in data.spec.axes)}")
    lines.append("")
    lines.append("## Test Results")
    lines.append("")
    lines.append("| Model | AUC | Acc | AOD | TPR gap/EOD | FPR gap | EOdds max | DP | CF flip | CF abs prob | Main abs logit | Pair abs logit | Proxy score | Proxy effect |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for result in results:
        b = result["test_behavior"]
        m = result["test_mechanism"]
        lines.append(
            "| "
            + str(result["label"])
            + " | "
            + " | ".join(
                [
                    fmt(float(b["auc"])),
                    fmt(float(b["accuracy"])),
                    fmt(float(b["aod_gap"])),
                    fmt(float(b["eod_gap"])),
                    fmt(float(b["fpr_gap"])),
                    fmt(float(b["eodds_max_gap"])),
                    fmt(float(b["dp_gap"])),
                    fmt(float(m["cf_flip_rate"])),
                    fmt(float(m["cf_abs_prob_mean"])),
                    fmt(float(m["main_abs_logit_mean"])),
                    fmt(float(m["pair_abs_mean"])),
                    fmt(float(m["proxy_score_gap_cond_mean"])),
                    fmt(float(m["proxy_effect_gap_mean"])),
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append("## Group-Threshold Postprocessing Diagnostic")
    lines.append("")
    lines.append("This uses validation-selected group thresholds after training. It is a behavior-only baseline/diagnostic; it does not change the learned interaction mechanism.")
    lines.append("")
    lines.append("| Model | Acc | AOD | EOD | DP |")
    lines.append("|---|---:|---:|---:|---:|")
    for result in results:
        b = result["test_behavior_group_thresholds"]
        lines.append(
            f"| {result['label']} | {fmt(float(b['accuracy']))} | {fmt(float(b['aod_gap']))} | {fmt(float(b['eod_gap']))} | {fmt(float(b['dp_gap']))} |"
        )
    lines.append("")
    lines.append("## Axis Interactions")
    lines.append("")
    axis_features = [axis.feature for axis in data.spec.axes]
    lines.append("| Model | " + " | ".join(axis_features) + " |")
    lines.append("|---|" + "|".join(["---:"] * len(axis_features)) + "|")
    for result in results:
        m = result["test_mechanism"]
        values = [fmt(float(m[f"pair_abs_mean__{feature}"])) for feature in axis_features]
        lines.append(f"| {result['label']} | " + " | ".join(values) + " |")
    lines.append("")
    lines.append("## Proxy Effect Gaps")
    lines.append("")
    lines.append("Each cell is the label-conditioned group gap in the model's logit effect of toggling that context axis.")
    lines.append("")
    lines.append("| Model | " + " | ".join(axis_features) + " |")
    lines.append("|---|" + "|".join(["---:"] * len(axis_features)) + "|")
    for result in results:
        m = result["test_mechanism"]
        values = [fmt(float(m[f"proxy_effect_gap_cond__{feature}"])) for feature in axis_features]
        lines.append(f"| {result['label']} | " + " | ".join(values) + " |")
    lines.append("")
    lines.append("## Specification Residuals")
    lines.append("")
    lines.append("Boundary residuals are measured on examples whose predicted probability is near the decision threshold. Slice residuals aggregate worst low/high context slices across the interaction axes.")
    lines.append("")
    lines.append("| Model | Boundary frac | Boundary score | Boundary effect | Slice score mean | Slice score max | Slice effect mean | Slice effect max |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for result in results:
        m = result["test_mechanism"]
        lines.append(
            f"| {result['label']} | "
            f"{fmt(float(m.get('boundary_fraction', float('nan'))))} | "
            f"{fmt(float(m.get('boundary_proxy_score_gap_cond_mean', float('nan'))))} | "
            f"{fmt(float(m.get('boundary_proxy_effect_gap_mean', float('nan'))))} | "
            f"{fmt(float(m.get('slice_proxy_score_gap_mean', float('nan'))))} | "
            f"{fmt(float(m.get('slice_proxy_score_gap_max', float('nan'))))} | "
            f"{fmt(float(m.get('slice_proxy_effect_gap_mean', float('nan'))))} | "
            f"{fmt(float(m.get('slice_proxy_effect_gap_max', float('nan'))))} |"
        )
    lines.append("")
    lines.append("## Run Configuration")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(to_jsonable(vars(args)), indent=2, sort_keys=True, default=str))
    lines.append("```")
    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.torch_threads)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    all_results: list[dict[str, object]] = []
    data_for_report: PreparedData | None = None
    for seed in args.seeds:
        data = load_data(args, seed)
        data_for_report = data
        seed_dir = args.output_dir / data.name / f"seed_{seed}"
        write_json(seed_dir / "dataset_summary.json", dataset_summary_payload(data))
        for model_name in args.models:
            result = train_one_model(model_name, data, seed, device, args, seed_dir / args.artifact_dir_name)
            all_results.append(result)
            write_json(seed_dir / f"{model_name}.json", result)
    assert data_for_report is not None
    write_json(args.output_dir / data_for_report.name / "all_results.json", {"results": all_results})
    write_report(args.output_dir / data_for_report.name / args.report_name, data_for_report, all_results, args)


if __name__ == "__main__":
    main()

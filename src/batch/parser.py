import copy
import fnmatch
import itertools
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.core.errors import ConfigError
from src.utils.io import load_yaml


@dataclass
class BatchCase:
    case_id: str
    data_path: str
    model_path: str
    data_selectors: Dict[str, Any] = field(default_factory=dict)
    model_selectors: Dict[str, Any] = field(default_factory=dict)
    run_name: str = ""
    output_dir: str = ""
    checkpoint_path: Optional[str] = None
    split: str = "test"
    extra_overrides: List[str] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BatchConfig:
    name: str
    task_name: str
    base_data: Optional[str]
    base_model: Optional[str]
    cases: List[BatchCase]
    output_root: str
    execution_config: Dict[str, Any] = field(default_factory=dict)


def _safe_format(template: str, values: Dict[str, Any]) -> str:
    """Format string using available keys, leaving unmatched placeholders intact."""
    result = template
    for k, v in values.items():
        placeholder = f"{{{k}}}"
        if placeholder in result:
            result = result.replace(placeholder, str(v))
    return result


def _normalize_selector_values(val: Any) -> List[Any]:
    if isinstance(val, list):
        return val
    return [val]


def expand_batch_config(
    config_or_path: Any,
    only: Optional[str] = None,
    output_dir_override: Optional[str] = None,
    cli_overrides: Optional[List[str]] = None,
) -> BatchConfig:
    """Load batch config and expand into a unified list of concrete BatchCases.
    Supports both Strategy 1 (matrix / Cartesian product grid) and Strategy 2 (cases / List of Dict).
    """
    if isinstance(config_or_path, str):
        if not os.path.isfile(config_or_path):
            raise ConfigError(f"Batch config file not found: {config_or_path}")
        raw = load_yaml(config_or_path) or {}
    elif isinstance(config_or_path, dict):
        raw = copy.deepcopy(config_or_path)
    else:
        raise ConfigError(f"Invalid batch config input type: {type(config_or_path)}")

    meta = raw.get("meta", {})
    batch_name = meta.get("name", "batch_run")
    task_name = meta.get("task_name", meta.get("task", "anomaly"))

    base_data = raw.get("base_data")
    base_model = raw.get("base_model")

    execution = raw.get("execution", {})
    output_root = output_dir_override or execution.get("output_root", os.path.join("outputs", batch_name))
    default_split = execution.get("split", "test")
    run_name_pattern = execution.get("run_name_pattern")
    checkpoint_pattern = execution.get("checkpoint_pattern")

    matrix_def = raw.get("matrix", {})
    explicit_cases = raw.get("cases", raw.get("experiments", []))

    if not matrix_def and not explicit_cases:
        raise ConfigError(
            f"Batch config must specify either 'matrix' (grid) or 'cases' (list of dict): {batch_name}"
        )

    expanded_cases: List[BatchCase] = []

    # 1. Strategy 1: Grid matrix expansion (Cartesian product)
    if matrix_def:
        data_matrix = matrix_def.get("data", {})
        model_matrix = matrix_def.get("model", {})

        data_keys = list(data_matrix.keys())
        data_values_list = [_normalize_selector_values(data_matrix[k]) for k in data_keys]

        model_keys = list(model_matrix.keys())
        model_values_list = [_normalize_selector_values(model_matrix[k]) for k in model_keys]

        data_combos = list(itertools.product(*data_values_list)) if data_values_list else [()]
        model_combos = list(itertools.product(*model_values_list)) if model_values_list else [()]

        for d_combo in data_combos:
            d_sel = {data_keys[i]: d_combo[i] for i in range(len(d_combo))}
            for m_combo in model_combos:
                m_sel = {model_keys[i]: m_combo[i] for i in range(len(m_combo))}

                # Extract context tokens for naming
                context_tokens = {
                    "task": task_name,
                    "batch_name": batch_name,
                }
                for k, v in d_sel.items():
                    context_tokens[f"data_{k}"] = v
                    context_tokens[k] = v
                for k, v in m_sel.items():
                    context_tokens[f"model_{k}"] = v
                    if k not in context_tokens:
                        context_tokens[k] = v

                # Model name extraction from path
                if base_model:
                    m_stem = os.path.splitext(os.path.basename(base_model))[0]
                    context_tokens["model_name"] = m_stem

                # Generate case ID & Run Name
                id_parts = []
                if "category" in d_sel:
                    id_parts.append(str(d_sel["category"]))
                if "backbone" in m_sel:
                    id_parts.append(str(m_sel["backbone"]))
                elif "size" in m_sel:
                    id_parts.append(str(m_sel["size"]))
                elif not id_parts:
                    id_parts.append(f"case_{len(expanded_cases):03d}")

                case_id = "_".join(id_parts)
                context_tokens["case_id"] = case_id

                if run_name_pattern:
                    run_name = _safe_format(run_name_pattern, context_tokens)
                else:
                    parts = [task_name]
                    if "model_name" in context_tokens:
                        parts.append(context_tokens["model_name"])
                    if "backbone" in m_sel:
                        parts.append(str(m_sel["backbone"]))
                    elif "size" in m_sel:
                        parts.append(str(m_sel["size"]))
                    if "category" in d_sel:
                        parts.append(str(d_sel["category"]))
                    run_name = "_".join(parts)

                context_tokens["run_name"] = run_name
                case_dir = os.path.join(output_root, run_name)

                # Checkpoint resolution
                ckpt_path = None
                if checkpoint_pattern:
                    context_tokens["output_dir"] = case_dir
                    ckpt_path = _safe_format(checkpoint_pattern, context_tokens)
                else:
                    ckpt_path = os.path.join(case_dir, "checkpoints", "best.pth")

                case_overrides = list(cli_overrides or [])
                if execution.get("epochs") is not None:
                    case_overrides.append(f"train.epochs={execution['epochs']}")
                if execution.get("batch_size") is not None:
                    case_overrides.append(f"data.batch_size={execution['batch_size']}")
                if execution.get("device") is not None:
                    case_overrides.append(f"runtime.device={execution['device']}")
                if execution.get("seed") is not None:
                    case_overrides.append(f"runtime.seed={execution['seed']}")

                expanded_cases.append(
                    BatchCase(
                        case_id=case_id,
                        data_path=base_data,
                        model_path=base_model,
                        data_selectors=d_sel,
                        model_selectors=m_sel,
                        run_name=run_name,
                        output_dir=case_dir,
                        checkpoint_path=ckpt_path,
                        split=default_split,
                        extra_overrides=case_overrides,
                        meta=context_tokens,
                    )
                )

    # 2. Strategy 2: Explicit cases list (List of Dict)
    if explicit_cases:
        for idx, c_def in enumerate(explicit_cases):
            c_name = c_def.get("name", f"case_{idx:03d}")
            d_path = c_def.get("data", base_data)
            m_path = c_def.get("model", base_model)
            d_sel = c_def.get("data_selectors", {})
            m_sel = c_def.get("model_selectors", {})

            context_tokens = {
                "task": task_name,
                "case_id": c_name,
                "batch_name": batch_name,
            }
            for k, v in d_sel.items():
                context_tokens[f"data_{k}"] = v
                context_tokens[k] = v
            for k, v in m_sel.items():
                context_tokens[f"model_{k}"] = v
                if k not in context_tokens:
                    context_tokens[k] = v

            if m_path:
                m_stem = os.path.splitext(os.path.basename(m_path))[0]
                context_tokens["model_name"] = m_stem

            run_name = c_def.get("run_name")
            if not run_name:
                if run_name_pattern:
                    run_name = _safe_format(run_name_pattern, context_tokens)
                else:
                    run_name = f"{batch_name}_{c_name}"

            context_tokens["run_name"] = run_name
            case_dir = c_def.get("output_dir") or os.path.join(output_root, run_name)

            ckpt_path = c_def.get("checkpoint")
            if not ckpt_path and checkpoint_pattern:
                context_tokens["output_dir"] = case_dir
                ckpt_path = _safe_format(checkpoint_pattern, context_tokens)
            elif not ckpt_path:
                ckpt_path = os.path.join(case_dir, "checkpoints", "best.pth")

            case_overrides = list(cli_overrides or []) + list(c_def.get("overrides", []))
            if execution.get("epochs") is not None and not any(o.startswith("train.epochs=") for o in case_overrides):
                case_overrides.append(f"train.epochs={execution['epochs']}")
            if execution.get("batch_size") is not None and not any(o.startswith("data.batch_size=") for o in case_overrides):
                case_overrides.append(f"data.batch_size={execution['batch_size']}")
            if execution.get("device") is not None and not any(o.startswith("runtime.device=") for o in case_overrides):
                case_overrides.append(f"runtime.device={execution['device']}")
            if execution.get("seed") is not None and not any(o.startswith("runtime.seed=") for o in case_overrides):
                case_overrides.append(f"runtime.seed={execution['seed']}")

            expanded_cases.append(
                BatchCase(
                    case_id=c_name,
                    data_path=d_path,
                    model_path=m_path,
                    data_selectors=d_sel,
                    model_selectors=m_sel,
                    run_name=run_name,
                    output_dir=case_dir,
                    checkpoint_path=ckpt_path,
                    split=c_def.get("split", default_split),
                    extra_overrides=case_overrides,
                    meta=context_tokens,
                )
            )

    # Filter cases if `only` filter is provided
    if only:
        filters = [f.strip() for f in only.split(",") if f.strip()]
        filtered = []
        for case in expanded_cases:
            match = False
            for pat in filters:
                if fnmatch.fnmatch(case.case_id, f"*{pat}*") or fnmatch.fnmatch(case.run_name, f"*{pat}*"):
                    match = True
                    break
                # Check selector values as well
                for v in case.data_selectors.values():
                    if fnmatch.fnmatch(str(v), f"*{pat}*"):
                        match = True
                        break
                for v in case.model_selectors.values():
                    if fnmatch.fnmatch(str(v), f"*{pat}*"):
                        match = True
                        break
            if match:
                filtered.append(case)
        expanded_cases = filtered

    return BatchConfig(
        name=batch_name,
        task_name=task_name,
        base_data=base_data,
        base_model=base_model,
        cases=expanded_cases,
        output_root=output_root,
        execution_config=execution,
    )

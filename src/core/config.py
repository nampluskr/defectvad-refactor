import copy
import os
import re

import torch
import yaml

from src.core.errors import ConfigError
from src.core.registry import ADAPTERS, DATASETS, LOSSES, METRICS, MODELS, TRANSFORMS

TOP_LEVEL_KEYS = {
    "meta", "runtime", "data", "model", "loss", "metrics", "adapter", "optim", "train", "output",
}

# 'paths' is allowed but not required: only tasks whose _base.yaml declares a paths block (SPEC
# SS6.1) use it -- a task with no external assets needs no dataset_root/backbone_root (NFR-003).
ALLOWED_TOP_LEVEL_KEYS = TOP_LEVEL_KEYS | {"paths"}

MAX_INHERIT_DEPTH = 3

DEFAULT_LOCAL_CONFIG_PATH = "configs/local.yaml"

# rank 2 (SPEC SS6.2): env var name for each paths.<key>
PATH_ENV_VARS = {
    "dataset_root": "DATASET_DIR",
    "backbone_root": "BACKBONE_DIR",
}

PLACEHOLDER_PATTERN = re.compile(r"\$\{paths\.([a-zA-Z0-9_]+)\}")


def deep_merge(base, override):
    if not isinstance(base, dict) or not isinstance(override, dict):
        return copy.deepcopy(override)
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_raw_yaml(path):
    if not os.path.isfile(path):
        raise ConfigError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_and_merge_base(path, depth=0, visited=None):
    if visited is None:
        visited = set()
    if depth > MAX_INHERIT_DEPTH:
        raise ConfigError(f"_base inheritance depth exceeds {MAX_INHERIT_DEPTH} at '{path}'.")
    real_path = os.path.realpath(path)
    if real_path in visited:
        raise ConfigError(f"Circular _base reference detected at '{path}'.")
    visited = visited | {real_path}

    raw = load_raw_yaml(path)
    base_spec = raw.pop("_base", None)

    merged = {}
    if base_spec is not None:
        base_paths = base_spec if isinstance(base_spec, list) else [base_spec]
        for base_path in base_paths:
            resolved_base_path = os.path.join(os.path.dirname(path), base_path)
            if not os.path.isfile(resolved_base_path):
                resolved_base_path = base_path
            merged = deep_merge(merged, load_and_merge_base(resolved_base_path, depth + 1, visited))

    merged = deep_merge(merged, raw)
    return merged


def parse_set_arg(arg):
    if "=" not in arg:
        raise ConfigError(f"--set argument must be of form <dotted.key>=<yaml_value>, got '{arg}'")
    key, _, value_str = arg.partition("=")
    value = yaml.safe_load(value_str)
    return key.strip(), value


def apply_set_override(config, key, value):
    parts = key.split(".")
    node = config
    for part in parts[:-1]:
        node = descend(node, part, key)
    set_leaf(node, parts[-1], value, key)


def descend(node, part, full_key):
    if isinstance(node, list):
        if not part.lstrip("-").isdigit() or int(part) >= len(node):
            raise ConfigError(f"--set key '{full_key}' references nonexistent index '{part}'.")
        return node[int(part)]
    if isinstance(node, dict):
        if part not in node:
            raise ConfigError(f"--set key '{full_key}' references nonexistent key '{part}'.")
        return node[part]
    raise ConfigError(f"--set key '{full_key}' cannot descend into a scalar at '{part}'.")


def set_leaf(node, part, value, full_key):
    if isinstance(node, list):
        if not part.lstrip("-").isdigit() or int(part) >= len(node):
            raise ConfigError(f"--set key '{full_key}' references nonexistent index '{part}'.")
        node[int(part)] = value
        return
    if isinstance(node, dict):
        if part not in node:
            raise ConfigError(f"--set key '{full_key}' references nonexistent key '{part}'.")
        node[part] = value
        return
    raise ConfigError(f"--set key '{full_key}' cannot set leaf on a scalar at '{part}'.")


def apply_overrides(config, override_args):
    config = copy.deepcopy(config)
    for arg in override_args or []:
        key, value = parse_set_arg(arg)
        apply_set_override(config, key, value)
    return config


def cli_override_keys(override_args, prefix):
    """Dotted keys under `prefix.` (e.g. 'paths') present in --set arguments."""
    keys = set()
    for arg in override_args or []:
        key, _ = parse_set_arg(arg)
        head, _, rest = key.partition(".")
        if head == prefix and rest:
            keys.add(rest)
    return keys


def used_placeholder_keys(config):
    """paths.<key> names actually referenced by ${paths.<key>} somewhere in config, excluding
    the 'paths' block itself (which only declares values, never references them)."""
    keys = set()

    def scan(node, top_level_key=None):
        if top_level_key == "paths":
            return
        if isinstance(node, str):
            keys.update(PLACEHOLDER_PATTERN.findall(node))
        elif isinstance(node, dict):
            for k, v in node.items():
                scan(v, k)
        elif isinstance(node, list):
            for v in node:
                scan(v)

    scan(config)
    return keys


def resolve_paths(config, override_args=None, local_config_path=None):
    """Resolve config['paths'] using the SPEC SS6.2 priority (highest wins):
    1. CLI --set paths.<key>=... (already merged into config by apply_overrides)
    2. env vars DATASET_DIR / BACKBONE_DIR
    3. configs/local.yaml (machine-specific SSOT, gitignored)
    4. the paths block already present in config (dev-machine default from _base.yaml)
    Only keys actually referenced by a ${paths.<key>} placeholder elsewhere in this config are
    required to exist (NFR-003) -- a task's paths block may declare a root that a from-scratch
    model config never uses. Records which source won for each key in
    config['paths']['_source'] (NFR-002)."""
    base_paths = {k: v for k, v in config.get("paths", {}).items() if k != "_source"}
    required_keys = used_placeholder_keys(config) & set(base_paths)
    if not required_keys:
        # No paths.* placeholder is actually referenced in this config (e.g. a from-scratch
        # model, or a task with no external assets at all) -- nothing to resolve or validate.
        config = copy.deepcopy(config)
        config.pop("paths", None)
        return config

    local_config_path = local_config_path or DEFAULT_LOCAL_CONFIG_PATH
    cli_keys = cli_override_keys(override_args, "paths")

    local_paths = {}
    if os.path.isfile(local_config_path):
        local_paths = (load_raw_yaml(local_config_path) or {}).get("paths", {}) or {}

    resolved = {}
    source = {}
    for key in required_keys:
        default_value = base_paths[key]
        env_name = PATH_ENV_VARS.get(key)
        if key in cli_keys:
            value, origin = default_value, "cli"
        elif env_name and env_name in os.environ:
            value, origin = os.environ[env_name], "env"
        elif key in local_paths:
            value, origin = local_paths[key], "local_yaml"
        else:
            value, origin = default_value, "config_default"
        require(
            isinstance(value, str),
            f"paths.{key} must be a string path, got {value!r} (source: {origin})."
        )
        require(
            os.path.isdir(value),
            f"paths.{key} does not exist: {value} (source: {origin}). "
            f"Fix it via --set paths.{key}=<path>, env var {env_name}=<path>, or "
            f"'{local_config_path}' (paths.{key}: <path>)."
        )
        resolved[key] = value
        source[key] = origin

    config = copy.deepcopy(config)
    config["paths"] = resolved
    config["paths"]["_source"] = source
    return config


def interpolate(config):
    """Substitute ${paths.<key>} placeholders in every string value, using config['paths']
    (already resolved by resolve_paths). Must run after --set is applied (SPEC SS6.4)."""
    paths = {k: v for k, v in config.get("paths", {}).items() if k != "_source"}

    def substitute(value):
        if isinstance(value, str):
            def replace(match):
                key = match.group(1)
                if key not in paths:
                    raise ConfigError(
                        f"Unknown path placeholder '${{paths.{key}}}'. "
                        f"Available paths keys: {sorted(paths)}"
                    )
                return str(paths[key])
            return PLACEHOLDER_PATTERN.sub(replace, value)
        if isinstance(value, dict):
            return {k: substitute(v) for k, v in value.items()}
        if isinstance(value, list):
            return [substitute(v) for v in value]
        return value

    return substitute(config)


def resolve_config(path, override_args=None, local_config_path=None):
    config = load_and_merge_base(path)
    config = apply_overrides(config, override_args)
    config = resolve_paths(config, override_args, local_config_path)
    config = interpolate(config)
    return config


def require(condition, message):
    if not condition:
        raise ConfigError(message)


def validate_config(config, check_paths=True, check_registry=True, check_cuda=True):
    # 1. top-level keys
    keys = set(config.keys())
    missing = TOP_LEVEL_KEYS - keys
    unknown = keys - ALLOWED_TOP_LEVEL_KEYS
    require(not missing, f"Config is missing required top-level keys: {sorted(missing)}")
    require(not unknown, f"Config has undefined top-level keys: {sorted(unknown)}")

    # 2. required subkeys and basic types
    data = config["data"]
    require(
        isinstance(data.get("image_size"), list) and len(data["image_size"]) == 2
        and all(isinstance(v, int) for v in data["image_size"]),
        "data.image_size must be a list of two integers [H, W].",
    )
    require(isinstance(data.get("batch_size"), int), "data.batch_size must be an integer.")
    require("split" in data and "mode" in data["split"], "data.split.mode is required.")
    require(data["split"]["mode"] in ("file", "ratio"), "data.split.mode must be 'file' or 'ratio'.")

    train = config["train"]
    require(isinstance(train.get("epochs"), int), "train.epochs must be an integer.")
    require("monitor" in train and "metric" in train["monitor"] and "mode" in train["monitor"],
            "train.monitor.metric and train.monitor.mode are required.")
    require(train["monitor"]["mode"] in ("max", "min"), "train.monitor.mode must be 'max' or 'min'.")

    # 3. {name, params} registry membership
    if check_registry:
        require_named(config["data"].get("name"), DATASETS, "data.name")
        require_named(config["model"].get("name"), MODELS, "model.name")
        require_named(config["loss"].get("name"), LOSSES, "loss.name")
        require_named(config["adapter"].get("name"), ADAPTERS, "adapter.name")
        for i, metric in enumerate(config.get("metrics", [])):
            require_named(metric.get("name"), METRICS, f"metrics.{i}.name")
        transform = config["data"].get("transform", {})
        for split_name in ("train", "eval"):
            if split_name in transform:
                require_named(transform[split_name].get("name"), TRANSFORMS,
                               f"data.transform.{split_name}.name")

    # 4. path existence
    if check_paths:
        require(os.path.exists(data["root"]), f"data.root does not exist: {data['root']}")
        if data["split"]["mode"] == "file":
            require(os.path.isfile(data["split"]["path"]),
                    f"data.split.path does not exist: {data['split']['path']}")
        weights_path = config["model"].get("params", {}).get("weights_path")
        if weights_path is not None:
            require(os.path.isfile(weights_path),
                    f"model.params.weights_path does not exist: {weights_path}")

    # 5. monitor metric must be declared
    metric_names = [m["name"] for m in config.get("metrics", [])]
    require(train["monitor"]["metric"] in metric_names,
            f"train.monitor.metric '{train['monitor']['metric']}' is not in metrics {metric_names}.")

    # 6. device availability
    if check_cuda and config["runtime"].get("device") == "cuda":
        require(torch.cuda.is_available(), "runtime.device is 'cuda' but torch.cuda.is_available() is False.")

    return config


def require_named(name, registry, field_label):
    require(name is not None, f"{field_label} is required.")
    require(name in registry.entries,
            f"{field_label} '{name}' is not registered in namespace '{registry.namespace}'. "
            f"Available: {registry.keys()}")

import copy
import os
import re
import yaml
import torch

from src.core.errors import ConfigError
from src.core.registry import ADAPTERS, DATASETS, LOSSES, METRICS, MODELS, TRANSFORMS
from src.utils.io import load_yaml

TOP_LEVEL_KEYS = {"meta", "data", "loss", "metrics", "adapter", "optim", "train", "output"}
ALLOWED_TOP_LEVEL_KEYS = TOP_LEVEL_KEYS | {"paths", "runtime", "model", "selectors", "derive"}

DEFAULT_LOCAL_CONFIG_PATH = os.path.join("configs", "local.yaml")
PATH_ENV_VARS = {
    "dataset_root": "DATASET_DIR",
    "backbone_root": "BACKBONE_DIR",
}
PLACEHOLDER_PATTERN = re.compile(r"\$\{paths\.([a-zA-Z0-9_]+)\}")


def load_raw_yaml(path):
    if not os.path.isfile(path):
        raise ConfigError(f"Config file not found: {path}")
    return load_yaml(path) or {}


def deep_merge(base, override):
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_and_merge_base(path):
    raw = load_raw_yaml(path)
    base_field = raw.get("_base")
    if not base_field:
        raw_copy = copy.deepcopy(raw)
        raw_copy.pop("_base", None)
        return raw_copy

    base_paths = [base_field] if isinstance(base_field, str) else list(base_field)
    merged_base = {}
    current_dir = os.path.dirname(path)

    for bp in base_paths:
        resolved_base_path = os.path.normpath(os.path.join(current_dir, bp))
        parent_config = load_and_merge_base(resolved_base_path)
        merged_base = deep_merge(merged_base, parent_config)

    raw_copy = copy.deepcopy(raw)
    raw_copy.pop("_base", None)
    return deep_merge(merged_base, raw_copy)


def parse_override_value(val_str):
    if not isinstance(val_str, str):
        return val_str
    try:
        return yaml.safe_load(val_str)
    except Exception:
        return val_str


def set_dotted_key(config, dotted_key, value):
    config = copy.deepcopy(config)
    parts = dotted_key.split(".")
    target = config
    for part in parts[:-1]:
        if part not in target or not isinstance(target[part], dict):
            target[part] = {}
        target = target[part]
    target[parts[-1]] = value
    return config


def get_dotted_key(config, dotted_key, default=None):
    parts = dotted_key.split(".")
    target = config
    for part in parts:
        if not isinstance(target, dict) or part not in target:
            return default
        target = target[part]
    return target


def apply_selectors(config, selectors_dict):
    if not selectors_dict or "selectors" not in config:
        return config

    config = copy.deepcopy(config)
    selectors_def = config.get("selectors", {})

    for key, val in selectors_dict.items():
        if key in selectors_def:
            selector_entry = selectors_def[key]
            if isinstance(selector_entry, dict):
                if val in selector_entry:
                    mapping = selector_entry[val]
                    if isinstance(mapping, dict):
                        for target_k, target_v in mapping.items():
                            config = set_dotted_key(config, target_k, target_v)
                else:
                    for target_k, template_v in selector_entry.items():
                        if isinstance(template_v, str):
                            substituted = template_v.replace("{value}", str(val))
                            config = set_dotted_key(config, target_k, substituted)
                        else:
                            config = set_dotted_key(config, target_k, template_v)
    return config


def apply_overrides(config, override_args):
    if not override_args:
        return config

    config = copy.deepcopy(config)
    if isinstance(override_args, dict):
        for k, v in override_args.items():
            config = set_dotted_key(config, k, v)
        return config

    for item in override_args:
        if "=" not in item:
            raise ConfigError(f"Invalid override format '{item}'. Expected KEY=VALUE.")
        key, val_str = item.split("=", 1)
        parsed_val = parse_override_value(val_str)
        config = set_dotted_key(config, key.strip(), parsed_val)
    return config


def used_placeholder_keys(config):
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
    base_paths = {k: v for k, v in config.get("paths", {}).items() if k != "_source"}
    required_keys = used_placeholder_keys(config) & set(base_paths)
    if not required_keys:
        config = copy.deepcopy(config)
        return config

    local_config_path = local_config_path or DEFAULT_LOCAL_CONFIG_PATH
    local_paths = {}
    if os.path.isfile(local_config_path):
        local_paths = (load_raw_yaml(local_config_path) or {}).get("paths", {}) or {}

    cli_keys = set()
    if override_args:
        if isinstance(override_args, dict):
            for k in override_args:
                if k.startswith("paths."):
                    cli_keys.add(k.split(".", 1)[1])
        elif isinstance(override_args, list):
            for item in override_args:
                if item.startswith("paths.") and "=" in item:
                    cli_keys.add(item.split("=")[0].split(".", 1)[1])

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
    if "paths" not in config:
        config["paths"] = {}
    config["paths"].update(resolved)
    config["paths"]["_source"] = source
    return config


def interpolate(config):
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


def resolve_config(
    data_path=None,
    model_path=None,
    config_path=None,
    data_selectors=None,
    model_selectors=None,
    cli_overrides=None,
    local_config_path=None,
):
    paths = [p for p in (data_path, model_path, config_path) if p]
    if not paths:
        raise ConfigError("At least one of data_path, model_path, or config_path must be provided.")

    merged = {}
    if data_path:
        data_cfg = load_and_merge_base(data_path)
        if data_selectors:
            data_cfg = apply_selectors(data_cfg, data_selectors)
        merged = deep_merge(merged, data_cfg)

    if model_path:
        model_cfg = load_and_merge_base(model_path)
        if model_selectors:
            model_cfg = apply_selectors(model_cfg, model_selectors)
        merged = deep_merge(merged, model_cfg)

    if config_path:
        self_contained = load_and_merge_base(config_path)
        merged = deep_merge(merged, self_contained)

    config = apply_overrides(merged, cli_overrides)
    config = resolve_paths(config, cli_overrides, local_config_path)
    config = interpolate(config)
    return config


def require(condition, message):
    if not condition:
        raise ConfigError(message)


def require_named(name, registry, field_label):
    require(name is not None, f"{field_label} is required.")
    require(
        name in registry.entries,
        f"{field_label} '{name}' is not registered in namespace '{registry.namespace}'. "
        f"Available: {registry.keys()}",
    )


def validate_config(config, check_paths=True, check_registry=True, check_cuda=True):
    keys = set(config.keys())
    missing = TOP_LEVEL_KEYS - keys
    unknown = keys - ALLOWED_TOP_LEVEL_KEYS
    require(not missing, f"Config is missing required top-level keys: {sorted(missing)}")
    require(not unknown, f"Config has undefined top-level keys: {sorted(unknown)}")

    data = config["data"]
    require(
        isinstance(data.get("image_size"), list)
        and len(data["image_size"]) == 2
        and all(isinstance(v, int) for v in data["image_size"]),
        "data.image_size must be a list of two integers [H, W].",
    )
    require(isinstance(data.get("batch_size"), int), "data.batch_size must be an integer.")
    require("split" in data and "mode" in data["split"], "data.split.mode is required.")
    require(
        data["split"]["mode"] in ("file", "ratio"),
        "data.split.mode must be 'file' or 'ratio'.",
    )

    train = config["train"]
    require(isinstance(train.get("epochs"), int), "train.epochs must be an integer.")
    require(
        "monitor" in train and "metric" in train["monitor"] and "mode" in train["monitor"],
        "train.monitor.metric and train.monitor.mode are required.",
    )
    require(
        train["monitor"]["mode"] in ("max", "min"),
        "train.monitor.mode must be 'max' or 'min'.",
    )

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
                require_named(
                    transform[split_name].get("name"),
                    TRANSFORMS,
                    f"data.transform.{split_name}.name",
                )

    if check_paths:
        require(os.path.exists(data["root"]), f"data.root does not exist: {data['root']}")
        if data["split"]["mode"] == "file":
            require(
                os.path.isfile(data["split"]["path"]),
                f"data.split.path does not exist: {data['split']['path']}",
            )
        weights_path = config["model"].get("params", {}).get("weights_path")
        if weights_path is not None:
            require(
                os.path.isfile(weights_path),
                f"model.params.weights_path does not exist: {weights_path}",
            )

    metric_names = [m["name"] for m in config.get("metrics", [])]
    require(
        train["monitor"]["metric"] in metric_names,
        f"train.monitor.metric '{train['monitor']['metric']}' is not in metrics {metric_names}.",
    )

    if check_cuda and config.get("runtime", {}).get("device") == "cuda":
        require(
            torch.cuda.is_available(),
            "runtime.device is 'cuda' but torch.cuda.is_available() is False.",
        )

    return config

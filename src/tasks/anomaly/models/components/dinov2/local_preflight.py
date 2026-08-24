from src.core.errors import LocalAssetError
from .dinov2_loader import DinoV2Loader


def require_local_dinov2_weight(cache_dir, encoder_name):
    """Raise ``LocalAssetError`` if the local weight file ``DinoV2Loader`` would
    request for ``encoder_name`` under ``cache_dir`` does not exist.

    ``DinoV2Loader._load_weights`` falls through to ``_download_weights`` on a
    miss (CON-003/CON-004 violation). ``weights_path`` in this project's model
    configs only pins one specific sibling file to satisfy the common config
    validator's ``os.path.isfile`` check -- the directory is what a factory
    actually points ``DinoV2Loader`` at, so a caller changing ``encoder_name``
    without also updating ``weights_path`` (e.g. via ``--set``) could otherwise
    reach that download path silently. Reuses ``DinoV2Loader``'s own (private)
    name-parsing/path-resolution methods rather than re-deriving the naming
    convention, so this preflight cannot drift from what the loader itself will
    request.
    """
    loader = DinoV2Loader(cache_dir=cache_dir)
    model_type, architecture, patch_size = loader._parse_name(encoder_name)
    weight_path = loader._get_weight_path(model_type, architecture, patch_size)
    if not weight_path.is_file():
        raise LocalAssetError(
            f"DINOv2 encoder '{encoder_name}' requires local weights at "
            f"{weight_path}, which does not exist. Network download is disabled."
        )

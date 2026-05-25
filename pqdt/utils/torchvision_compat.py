_TORCHVISION_LIBRARY = None


def ensure_torchvision_nms_schema():
    """Define the torchvision::nms schema when torchvision was installed without it."""
    global _TORCHVISION_LIBRARY

    import warnings

    warnings.filterwarnings(
        "ignore",
        message="Failed to load image Python extension:.*",
        category=UserWarning,
        module="torchvision.io.image",
    )

    try:
        import torch
    except ImportError:
        return

    try:
        import torchvision  # noqa: F401
        return
    except RuntimeError as exc:
        if "torchvision::nms" not in str(exc):
            raise

        import sys

        for module_name in list(sys.modules):
            if module_name == "torchvision" or module_name.startswith("torchvision."):
                sys.modules.pop(module_name, None)

    try:
        torch._C._dispatch_has_kernel_for_dispatch_key("torchvision::nms", "Meta")
        return
    except RuntimeError:
        pass

    try:
        _TORCHVISION_LIBRARY = torch.library.Library("torchvision", "DEF")
        _TORCHVISION_LIBRARY.define("nms(Tensor dets, Tensor scores, float iou_threshold) -> Tensor")
    except RuntimeError as exc:
        if "already defined" not in str(exc):
            raise

from importlib import import_module


MODELS = {
    "pqdt": "pqdt.models.completion:PQCompletionModel",
}


def _load_class(spec):
    module_name, class_name = spec.split(":")
    return getattr(import_module(module_name), class_name)


def build_model(cfg):
    try:
        model_cls = _load_class(MODELS[cfg.model.name])
    except KeyError as exc:
        valid = ", ".join(sorted(MODELS))
        raise ValueError(f"Unknown model '{cfg.model.name}'. Available models: {valid}") from exc
    return model_cls(cfg.model, total_epochs=cfg.trainer.max_epochs)


def __getattr__(name):
    if name == "PQCompletionModel":
        return _load_class("pqdt.models.completion:PQCompletionModel")
    raise AttributeError(name)


__all__ = ["MODELS", "PQCompletionModel", "build_model"]

from importlib import import_module


TASKS = {
    "restoration": "pqdt.tasks.restoration:RestorationLitModule",
}


def _load_class(spec):
    module_name, class_name = spec.split(":")
    return getattr(import_module(module_name), class_name)


def build_task(cfg):
    task = cfg.task
    name = task if isinstance(task, str) else getattr(task, "name")
    try:
        task_cls = _load_class(TASKS[name])
    except KeyError as exc:
        valid = ", ".join(sorted(TASKS))
        raise ValueError(f"Unknown task '{name}'. Available tasks: {valid}") from exc
    return task_cls(cfg)


def __getattr__(name):
    if name == "RestorationLitModule":
        return _load_class("pqdt.tasks.restoration:RestorationLitModule")
    raise AttributeError(name)


__all__ = ["RestorationLitModule", "TASKS", "build_task"]

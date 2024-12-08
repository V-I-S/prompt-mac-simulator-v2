import importlib
from typing import Any


def class_by_name(module_path: str, classname: str) -> Any:
    module = importlib.import_module(module_path + '.' + _module_of_class(classname))
    return getattr(module, classname)


def instantiate(module: str, classname: str, *args) -> Any:
    return class_by_name(module, classname)(*args)


def _module_of_class(classname: str) -> str:
    return classname[0].lower() + classname[1:]

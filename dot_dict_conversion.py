from typing import Any, Union


class DotDict(dict):
    """dot notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def to_dict(item: Union[object, dict]):
    if (isinstance(item, dict)):
        return item

    if (hasattr(item, '__dict__')):
        return vars(item)

    if ('__dict__' in item):
        return vars(item)

    return item


def to_dot_dict(item: Union[object, dict]) -> DotDict:

    item_dict: dict[str, Any] = to_dict(item)

    if (item_dict):
        return DotDict(item_dict)

    raise Exception("Failed converting item to DotDict instance: " + str(item))


def check_set_or_set_defaults(target_options: dict[str, Any], defaults_dict: dict[str, Any]) -> dict[str, Any]:
    for option_key in defaults_dict:
        if (option_key not in target_options or target_options[option_key] == None):
            target_options[option_key] = defaults_dict[option_key]

    return target_options


def to_dict_with_defaults(item: Union[object, dict], defaults_dict: dict[str, Any]) -> DotDict:

    item_dict: dict[str, Any] = to_dict(item)
    check_set_or_set_defaults(item_dict, defaults_dict)
    return item_dict


def to_dot_dict_with_defaults(item: Union[object, dict], defaults_dict: dict[str, Any]) -> DotDict:

    item_dict: dict[str, Any] = to_dict_with_defaults(item, defaults_dict)
    return DotDict(item_dict)

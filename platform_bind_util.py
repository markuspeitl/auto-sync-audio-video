from typing import Any


def get_platform_store_value(key: str, platform_values_store: dict[str, dict[str, Any]]) -> Any:
    if (not key in platform_values_store):
        raise Exception(f"Can not access key of platform_values_store: {key} -> is not defined")

    return get_platform_value(platform_values_store[key])


def get_platform_value(platform_value_options: dict[str, Any]) -> Any:
    from sys import platform

    current_plaform_value = None
    if platform == "linux" or platform == "linux2":
        current_plaform_value = platform_value_options['linux']

    elif platform == "darwin":
        current_plaform_value = platform_value_options['osx']

    elif platform == "win32":
        current_plaform_value = platform_value_options['windows']

    if (not current_plaform_value):
        raise Exception(f"No value registered for current platform: {platform}")

    return current_plaform_value

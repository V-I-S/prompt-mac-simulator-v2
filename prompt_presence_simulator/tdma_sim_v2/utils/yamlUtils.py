import yaml


def read_yaml(location: str) -> dict:
    with open(location) as yaml_config:
        dict_content = yaml.safe_load(yaml_config)
    return dict_content


def flatten_yaml(content: dict) -> dict:
    flat_dict = {}
    for origin_key, origin_value in content.items():
        if type(origin_value) is dict:
            for key, value in flatten_yaml(origin_value).items():
                flat_dict[f'{origin_key}.{key}'] = value
        else:
            flat_dict[origin_key] = origin_value
    return flat_dict

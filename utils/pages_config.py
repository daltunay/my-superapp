import typing as t
import yaml
import utils


def load_pages_config() -> t.Dict:
    with open("pages/pages_config.yaml", "r") as file:
        config = yaml.safe_load(file)
    return config


def load_page_config(config, file: str) -> None:
    logger = utils.CustomLogger(file)

    def set_recursive(section, keys, page_config):
        for key, value in section.items():
            if isinstance(value, dict):
                # If the value is a dictionary, go deeper recursively
                set_recursive(value, keys + [key])
            else:
                # Set the global variable using keys as the path
                page_config[key] = value

    path_keys = file.split("my-app/pages/")[1].split("/")
    section = config
    for path_key in path_keys:
        section = section.get(path_key, {})
    page_config = {}
    set_recursive(section, path_keys, page_config)
    logger.info(page_config)
    return page_config

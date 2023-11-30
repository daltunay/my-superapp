import utils

utils.load_secrets()

CONFIG = utils.load_pages_config()
page_config = utils.load_page_config(CONFIG, __file__)
for key, value in page_config.items():
    globals()[key] = value

__all__ = ["CONFIG"]

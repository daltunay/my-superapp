import utils
from pages import CONFIG

page_config = utils.load_page_config(CONFIG, __file__)
for key, value in page_config.items():
    globals()[key] = value


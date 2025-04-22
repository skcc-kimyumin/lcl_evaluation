import logging.config
import os
from functools import lru_cache
from pathlib import Path

import yaml
from core.config import get_setting

settings = get_setting()

log_dir = settings.DATA_PATH + "/logs/" + settings.APP_NAME
if not os.path.exists(log_dir):
    os.makedirs(log_dir, exist_ok=True)

# Load the config file
logging_file = os.path.join(os.path.dirname(__file__), "logging_config.yaml")
with open(logging_file, "rt") as f:
    config = yaml.safe_load(f.read())
    config["handlers"]["file"]["filename"] = log_dir + f"/{settings.APP_NAME}.log"


@lru_cache()
def get_logging():
    # Configure the logging module with the config file
    logging.config.dictConfig(config)
    logging.getLogger("sqlalchemy.engine").setLevel(settings.LOG_LEVEL)
    logging.getLogger("opensearch").setLevel(settings.LOG_LEVEL)

    # Get a logger object
    logger = logging.getLogger(settings.APP_NAME)
    logger.setLevel(settings.LOG_LEVEL)
    return logger


# Log some messages
# logger = get_logging()
# logger.debug('This is a debug message')
# logger.info('This is an info message')
# logger.warning('This is a warning message')
# logger.error('This is an error message')
# logger.critical('This is a critical message')

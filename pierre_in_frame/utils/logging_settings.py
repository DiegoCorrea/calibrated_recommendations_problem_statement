import json
import logging
import os

from settings.path_dir_file import PathDirFile


def setup_logging(
        default_path='./logging.json',
        default_level=logging.DEBUG,
        env_key='LOG_CFG',
        log_info='info.log',
        log_error='errors.log',
        save_path=PathDirFile.LOG_DIR
):
    """Setup logging configuration

    """
    path = default_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    value = os.getenv(env_key, None)
    erro_file_name = save_path + str(log_error)
    f = open(erro_file_name, "w")
    f.write("")
    f.close()
    info_file_name = save_path + str(log_info)
    f = open(info_file_name, "w")
    f.write("")
    f.close()
    if value:
        path = value
    if os.path.exists(path):
        with open(path, 'rt') as f:
            config = json.load(f)
            config["handlers"]["info_file_handler"]["filename"] = info_file_name
            config["handlers"]["error_file_handler"]["filename"] = erro_file_name
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)

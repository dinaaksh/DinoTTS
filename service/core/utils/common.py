from box.exceptions import BoxValueError
import yaml
from core.utils.logger import logger
from box import ConfigBox
from pathlib import Path
from typing import Union

def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """reads yaml file and returns

    Args:
        path_to_yaml (str): path like input

    Raises:
        ValueError: if yaml file is empty
        e: empty file

    Returns:
        ConfigBox: ConfigBox type
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("yaml file is empty")
    except Exception as e:
        raise e
    
def write_yaml(path: Path, data: Union[dict, ConfigBox]):
    with open(path, "w") as file:
        yaml.dump(dict(data), file)  
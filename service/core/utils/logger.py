import logging
import os

# Setup log directory
log_dir="logs"
os.makedirs(log_dir, exist_ok=True)
log_file_path=os.path.join(log_dir, "running_logs.log")

# Define log format
log_format="[%(asctime)s: %(levelname)s: %(module)s: %(message)s]"
formatter=logging.Formatter(log_format)

# Create logger
logger=logging.getLogger("cloner")
logger.setLevel(logging.DEBUG)

# Avoid adding handlers multiple times (especially in Jupyter/Notebooks)
if not logger.handlers:
    # File handler
    file_handler=logging.FileHandler(log_file_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console handler
    stream_handler=logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

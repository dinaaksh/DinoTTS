import os
import mlflow
import shutil
from pathlib import Path
from core.utils.logger import logger
from core.utils.common import read_yaml,write_yaml
from core.utils.constants import *
from mlflow.tracking import MlflowClient

MLFLOW_TRACKING_URI=os.getenv('MLFLOW_TRACKING_URI')
MLFLOW_TRACKING_USERNAME=os.getenv('MLFLOW_TRACKING_USERNAME')
MLFLOW_TRACKING_PASSWORD=os.getenv('MLFLOW_TRACKING_PASSWORD')

class LoadModel:
    @staticmethod
    def get_model():
        try:
            config=read_yaml(CONFIG_FILE_PATH)
            model_name="vits_ljspeech"
            version="2"
            model_uri=f"models:/{model_name}/{version}"
            logger.info(f"Model URI: {model_uri}")

            client=MlflowClient()
            model_version=client.get_model_version(name=model_name, version=version)
            run_id=model_version.run_id
            logger.info(f"Run ID: {run_id}")

            run=client.get_run(run_id)
            artifact_uri_root=run.info.artifact_uri 
            logger.info(f"Resolved artifact URI root: {artifact_uri_root}")

            target_dir=Path("mlflow_artifacts") / f"{model_name}_v{version}"
            model_path=target_dir / "optimized_model.pth"
            config_path=target_dir / "config.json"

            model_exists=model_path.exists()
            config_exists=config_path.exists()

            if not model_exists or not config_exists:
                logger.info("Downloading full model artifacts")

                temp_dir=Path(mlflow.artifacts.download_artifacts(artifact_uri=model_uri))

                config_uri = f"{artifact_uri_root}/config.json"
                logger.info(f"Config URI: {config_uri}")
                temp_config_path = Path(mlflow.artifacts.download_artifacts(artifact_uri=config_uri))

                if target_dir.exists():
                    shutil.rmtree(target_dir)
                target_dir.mkdir(parents=True)

                for item in temp_dir.iterdir():
                    shutil.move(str(item), str(target_dir))

                shutil.copy(str(temp_config_path), str(config_path))

            else:
                logger.info("Model and config.json already exist. Skipping download.")

            if not model_path.exists():
                raise FileNotFoundError(f"Checkpoint file not found: {model_path}")
            if not config_path.exists():
                raise FileNotFoundError(f"Config file not found: {config_path}")

            config["MODEL_PATH"]=model_path.as_posix()
            config["CONFIG_PATH"]=config_path.as_posix()
            write_yaml(CONFIG_FILE_PATH, dict(config))
            return model_path.as_posix(),config_path.as_posix()

        except Exception as e:
            logger.error(f"Error occurred: {e}")
            raise



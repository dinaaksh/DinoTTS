import mlflow
import torch
import shutil
from pathlib import Path
from cloner.utils.logger import logger

class GetModel():
    def get_model():
        try:
            model_name = "vits_ljspeech"
            version = 2
            model_uri = f"models:/{model_name}/{version}"
            logger.info(f"Model URI: {model_uri}")

            target_dir = Path("mlflow_artifacts") / f"{model_name}_v{version}"

            if target_dir.exists() and (target_dir / "optimized_model.pth").exists():
                logger.info("Model already exists locally, skipping download.")
            else:
                logger.info("Downloading model...")

                temp_dir = Path(mlflow.artifacts.download_artifacts(artifact_uri=model_uri))

                if target_dir.exists():
                    shutil.rmtree(target_dir)
                target_dir.mkdir(parents=True)

                for item in temp_dir.iterdir():
                    shutil.move(str(item), str(target_dir))

                logger.info(f"Model downloaded to: {target_dir}")

            model_path = target_dir / "optimized_model.pth" 
            if not model_path.exists():
                raise FileNotFoundError(f"Checkpoint file not found: {model_path}")

            return model_path

        except Exception as e:
            print(f"Error occurred: {e}")

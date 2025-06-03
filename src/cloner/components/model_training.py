from trainer import Trainer, TrainerArgs
from cloner.utils.mlflow_logger import MLFlowLogger
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import Vits
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from cloner.utils.common import read_yaml
from cloner.pipeline.stage_02_data_preprocessing import DataPreprocessor
from cloner.entity.config_entity import DataPreProcessConfig
from cloner.config.configuration import ModelTrainingConfig
from cloner.constants import *
from cloner.utils.logger import logger
import mlflow
from mlflow.tracking import MlflowClient
import os
import tempfile
import shutil


class   ModelConfig:
    def __init__(self,config: ModelTrainingConfig):
        self.config=config
        self.params=read_yaml(PARAMS_FILE_PATH)

        self.audio_config=self.get_audio_config()
        self.dataset_config=self.get_dataset_config()
        self.vits_config=self.get_vits_config()

        self._audio_processor=None
        self._tokenizer=None
        self._model=None
        self._trainer_instance=None
        self._train_samples=None
        self._eval_samples=None

    def get_audio_config(self):
        return self.params["audio_config"]

    def get_dataset_config(self):
        return BaseDatasetConfig(
            formatter=self.config.dataset_name,
            meta_file_train=self.config.metadata_path,
            path=self.config.dataset_path
        )

    def get_vits_config(self):
        params=self.params["model_config"]
        audio_config=self.audio_config
        dataset_config=self.dataset_config
        return VitsConfig(
            audio=audio_config,
            run_name=params["run_name"],
            batch_size=params["batch_size"],
            eval_batch_size=params["eval_batch_size"],
            batch_group_size=params["batch_group_size"],
            num_loader_workers=params["num_loader_workers"],
            num_eval_loader_workers=params["num_eval_loader_workers"],
            run_eval=params["run_eval"],
            test_delay_epochs=params["test_delay_epochs"],
            epochs=params["epochs"],
            text_cleaner=params["text_cleaner"],
            use_phonemes=params["use_phonemes"],
            phoneme_language=params["phoneme_language"],
            phoneme_cache_path=os.path.join(params["output_path"], "phoneme_cache"),
            compute_input_seq_cache=params["compute_input_seq_cache"],
            print_step=params["print_step"],
            print_eval=params["print_eval"],
            mixed_precision=params["mixed_precision"],
            output_path=params["output_path"],
            datasets=[dataset_config],
            cudnn_benchmark=params["cudnn_benchmark"],
        )

    def get_audio_processor(self):
        if self._audio_processor is None:
            data_preprocess_config=DataPreProcessConfig(
                root_dir=self.config.root_dir,
                processed_audio_dir="",  
                audio_path=""         
            )
            processor=DataPreprocessor(config=data_preprocess_config)
            self._audio_processor=processor.get_audio_processor()
        return self._audio_processor

    def get_tokenizer(self):
            if self._tokenizer is None:
                vits_config=self.vits_config
                tokenizer, config=TTSTokenizer.init_from_config(vits_config)
                self._tokenizer=tokenizer
            return self._tokenizer
    
    def get_data_split(self):
        if self._train_samples is None or self._eval_samples is None: 
            self._train_samples,self._eval_samples=load_tts_samples(
                self.dataset_config,
                eval_split=True,
                eval_split_max_size=self.vits_config.eval_split_max_size,
                eval_split_size=self.vits_config.eval_split_size,
            )
        return self._train_samples,self._eval_samples
    
    def get_model(self, checkpoint_path=None):
        if self._model is None:
            config=self.vits_config
            ap=self.get_audio_processor()
            tokenizer=self.get_tokenizer()
            self._model=Vits(config,ap,tokenizer,speaker_manager=None)
        return self._model
    
    def get_trainer(self, restore_path=None,use_mlflow=True):
        train_samples, eval_samples = self.get_data_split()
        model = self.get_model()
        
        trainer_args = TrainerArgs()
        trainer_args.restore_path = restore_path 

        mlflow_logger = self.get_mlflow_logger() if use_mlflow else None

        trainer_instance = Trainer(
            trainer_args,
            config=self.vits_config,
            output_path=self.config.output_dir,
            model=model,
            train_samples=train_samples,
            eval_samples=eval_samples,
            dashboard_logger=mlflow_logger,
            parse_command_line_args=False
        )
        return trainer_instance
    
    def get_mlflow_logger(self):
        params=self.params["model_config"]
        mlflow_uri=params["mlflow_tracking_uri"]
        model_name=params["run_name"]
        tags = {"mlflow.runName": f"Run for {model_name}"}
        
        mlflow_logger = MLFlowLogger(
            log_uri=mlflow_uri,
            model_name=model_name,
            tags=tags
        )

        keys = [
            "run_name","batch_size","eval_batch_size","batch_group_size","num_loader_workers","num_eval_loader_workers","run_eval","test_delay_epochs","epochs","text_cleaner",
            "use_phonemes","phoneme_language","compute_input_seq_cache","print_step","print_eval","mixed_precision","cudnn_benchmark"
        ]
        
        for key in keys:
            value=params.get(key)
            if isinstance(value, (str, int, float, bool)):
                mlflow_logger.client.log_param(mlflow_logger.run_id, key, value)

        return mlflow_logger
    
    def get_fit(self):
        restore_path = getattr(self.config, "restore_path", None)
        trainer = self.get_trainer(restore_path)
        self._trainer_instance = trainer
        trainer.fit()
        self.register_model()

    def get_latest_model_path(self, base_output_path):
        subdirs=[
            os.path.join(base_output_path, d)
            for d in os.listdir(base_output_path)
            if os.path.isdir(os.path.join(base_output_path, d))
        ]

        if not subdirs:
            logger.warning("No subdirectories found in output path.")
            return None

        latest_subdir=max(subdirs, key=os.path.getmtime)
        best_model_path=os.path.join(latest_subdir, "best_model.pth")

        if os.path.exists(best_model_path):
            return best_model_path
        else:
            logger.warning(f"'best_model.pth' not found in latest subdirectory: {latest_subdir}")
            return None
        
    def register_model(self, model_artifact_name="model"):

        if self._trainer_instance is None or not isinstance(self._trainer_instance.dashboard_logger, MLFlowLogger):
            logger.warning("Trainer not initialized or MLFlowLogger not used. Skipping model registration.")
            return

        mlflow_logger = self._trainer_instance.dashboard_logger
        if not isinstance(mlflow_logger, MLFlowLogger):
            print("MLFlowLogger not used. Skipping model registration.")
            return

        run_id=mlflow_logger.run_id
        client: MlflowClient=mlflow_logger.client
        base_output_path=self.vits_config.output_path
        model_path=self.get_latest_model_path(base_output_path)
        model_file_name="best_model.pth"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_model_path = os.path.join(temp_dir, model_file_name)
            shutil.copy(model_path, temp_model_path)
            logger.info("Uploading model to MLflow...")
            client.log_artifacts(run_id, temp_dir, artifact_path=model_artifact_name)

        model_uri = f"runs:/{run_id}/{model_artifact_name}"
        model_name = self.vits_config.run_name

        logger.info(f"Registering model '{model_name}' from URI: {model_uri}")
        try:
            registered_model = mlflow.register_model(model_uri=model_uri, name=model_name)
            logger.info(f"Model registered successfully: name={registered_model.name}, version={registered_model.version}")
        except Exception as e:
            logger.error(f"Failed to register model: {e}")


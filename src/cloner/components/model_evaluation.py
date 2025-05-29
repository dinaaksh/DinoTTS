import mlflow
import tempfile
import os
import torch
import torchaudio
import matplotlib.pyplot as plt
import seaborn as sns

from cloner import logger
from cloner.config.configuration import EvaluationConfig, ModelTrainingConfig
from cloner.components.model_training import ModelConfig


class AudioModelEvaluation:
    def __init__(self, eval_config: EvaluationConfig, train_config: ModelTrainingConfig):
        self.config = eval_config
        self.model_config = ModelConfig(config=train_config) 

    def log_into_mlflow(self):
        mlflow.set_tracking_uri(self.config.mlflow_uri)
        mlflow.set_experiment("AudioModelEvaluation") 

        with mlflow.start_run():
            # Log all given params
            mlflow.log_params(self.config.all_params)

            # Get trainer and run evaluation
            trainer = self.model_config.get_trainer(restore_path=self.config.model_path)
            eval_logs = trainer.eval()  # Dictionary of loss metrics

            # Log all evaluation metrics to MLflow
            mlflow.log_metrics(eval_logs)

            # Generate audio from sample text
            sample_text = "This is a test sentence for speech synthesis."
            model = self.model_config.get_model(checkpoint_path=self.config.model_path)
            audio_tensor = model.tts(sample_text)

            with tempfile.TemporaryDirectory() as tmpdir:
                # Save generated audio
                audio_path = os.path.join(tmpdir, "sample.wav")
                torchaudio.save(audio_path, torch.tensor(audio_tensor).unsqueeze(0), 22050)
                mlflow.log_artifact(audio_path, artifact_path="audio_outputs")

                # Save and log mel spectrogram image
                mel_path = os.path.join(tmpdir, "mel_spectrogram.png")
                self._save_mel_spectrogram(audio_tensor, mel_path)
                mlflow.log_artifact(mel_path, artifact_path="mel_spectrograms")

                # Log the model checkpoint used for evaluation
                mlflow.log_artifact(self.config.model_path, artifact_path="checkpoints")

        logger.info("Model evaluation and logging to MLflow completed.")

    def _save_mel_spectrogram(self, waveform, output_path):
        plt.figure(figsize=(10, 4))
        spec = torchaudio.transforms.MelSpectrogram(sample_rate=22050)(torch.tensor(waveform))
        spec_db = torchaudio.transforms.AmplitudeToDB()(spec)
        sns.heatmap(spec_db.log2()[0], cmap="viridis")
        plt.title("Mel Spectrogram")
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

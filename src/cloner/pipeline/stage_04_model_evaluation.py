from cloner.config.configuration import ConfigurationManager
from cloner.components.model_evaluation import AudioModelEvaluation
from cloner import logger

STAGE_NAME="Model Evaluation"

class ModelEvaluationPipeline:
    def __init__(self):
        pass

    def main(self):
            config=ConfigurationManager()
            model_training_config = config.get_model_training_config()
            model_evaluation_config=config.get_model_evaluation_config()
            model_evaluation = AudioModelEvaluation(
                eval_config=model_evaluation_config,
                train_config=model_training_config
            )
            model_evaluation.log_into_mlflow()

if __name__=='__main__':
    try:
        logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
        obj=ModelEvaluationPipeline()
        obj.main()
        logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<\n\n")
    except Exception as e:
        logger.exception(e)
        raise e

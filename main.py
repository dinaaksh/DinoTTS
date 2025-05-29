from cloner import logger 
from cloner.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from cloner.pipeline.stage_02_data_preprocessing import DataPreprocessingPipeline
from cloner.pipeline.stage_03_model_training import ModelConfigurationPipeline
from cloner.pipeline.stage_04_model_evaluation import ModelEvaluationPipeline

if __name__ == '__main__': 
    # STAGE_NAME="Data Ingestion Stage"
    # try:
    #     logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
    #     data_ingestion=DataIngestionTrainingPipeline()
    #     data_ingestion.main()
    #     logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<\n\n")
    # except Exception as e:
    #     logger.exception(e)
    #     raise e

    # STAGE_NAME="Data Preprocessing Stage"
    # try:
    #     logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
    #     data_preprocessing=DataPreprocessingPipeline()
    #     data_preprocessing.main()
    #     logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<\n\n")
    # except Exception as e:
    #     logger.exception(e)
    #     raise e 

    # STAGE_NAME="Model Implementation Stage"
    # try:
    #     logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
    #     model_configuration=ModelConfigurationPipeline()
    #     model_configuration.main()
    #     logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<\n\n")
    # except Exception as e:
    #     logger.exception(e)
    #     raise e 
    
    STAGE_NAME="Model Evaluation Stage"
    try:
        logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
        model_configuration=ModelEvaluationPipeline()
        model_configuration.main()
        logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<\n\n")
    except Exception as e:
        logger.exception(e)
        raise e

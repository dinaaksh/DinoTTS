from core.stages import stage_01,stage_03
from core.utils.logger import logger

def run_pipeline():

    STAGE_NAME="Model Downloading Stage"
    try:
        logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
        model_path,config_path=stage_01.LoadModel.get_model()
        logger.info(f"Model Path: {model_path}")
        logger.info(f"Config Path: {config_path}")  
        logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<\n\n")
    except Exception as e:
        logger.exception(e)
        raise e 
    
    STAGE_NAME=" Voice Cloning Stage"
    try:
        logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
        stage_03.tts.clone()
        logger.info("Output stored in output.wav in local directory.")
        logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<\n\n")
    except Exception as e:
        logger.exception(e)
        raise e 

if __name__ == '__main__':
    run_pipeline()

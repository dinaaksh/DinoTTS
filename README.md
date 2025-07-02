artifacts->all the outputs stored

config/config.yaml file-> stores the input and output paths necessary

logs/running_logs.log-> stores the logs of active run

research->1)01_data_ingestion.ipynb, test run of data ingestion phase
	  2)02_data_preprocess.ipynb, test run of preprocessing phase
	  3)03_model_setup.ipynb, test run of model configuration and training phase
	  4)testing.ipynb, contains the official code of coqui vits train.py 

src/cloner->main directory of the model

templates->html files for frontend

TTS->github repository of coqui TTS

main.py->function calls to pipeline

dvc.yaml->function calls to pipeline in a dvc environment

params.yaml->contains the parameter configuration for audio processor and model parameters

setup.py,requirements.txt->pre-requisite packages and modules

template.py->creates directories

COMPONENTS->

#use TTS_PATH=<your_path> if error locating directory

1)data_ingestion.py->

'DataIngestion' class built on 'DataIngestionConfig' entity
 contains functions: download_file(), extract_zip_file() 
#download_file function currently supports only google drive

2)data_preprocessing.py->

'DataPreprocessor' class built on 'DataPreProcessConfig' entity
 contains functions: get_audio_processor(), melspectogram(self,audio_path), process_audio()

3)model_training.py->

'ModelConfig' class built on 'ModelTrainingConfig' entity
 contains functions: get_audio_config(), get_vits_config(), get_audio_processor(), get_tokenizer(), get_data_split(), get_model(self,checkpoint_path=None), get_trainer(self,restore_path=None), load_model_from_checkpoint(self,restore_path), get_fit()


src/cloner/config/configuration.py->stores all return types of entities

src/cloner/constants->contains path to params.yaml and config.yaml

src/cloner/entity/config_entity.py->stores all entities

src/cloner/pipeline-> 1)stage_01_data_ingestion.py
		      2)stage_02_data_preprocessing.py
		      3)stage_03_model_training.py
	 
src/cloner/utils/common.py->contains functions which may be needed through code (e.g. read yaml)
src/cloner/_init_.py->contains logger

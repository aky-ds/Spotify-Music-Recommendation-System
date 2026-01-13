from src.components.Data_ingestion import DataIngestion
from src.components.Data_Transformation import DataTransformation
from src.components.Model_trainer import ModelTrainer
from src.components.Model_evaluation import ModelEvaluation
from src.exception import CustomException
from src.logger.logger import logging
def start_pipeline():
    try:
      logging.info("Training Pipeline has been Started")
      data_ingestion_obj=DataIngestion()
      dataframe=data_ingestion_obj.Instantiate_dataingestion()
      data_transformation_obj=DataTransformation()
      transformed_data = data_transformation_obj.instantiate_data_transformer(dataframe)
      model_trainer_obj=ModelTrainer()
      model_trainer_obj.train(transformed_data)
      model_evaluation=ModelEvaluation("artifacts/best_clustering_model.pkl")
      print(model_evaluation.evaluate(transformed_data))
      logging.info("Training Pipeline has been completed")
    except Exception as e:
       raise CustomException(e)



if __name__=="__main__":
   start_pipeline()
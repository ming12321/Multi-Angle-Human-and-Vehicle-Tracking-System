from comet_ml import start
from comet_ml.integration.pytorch import log_model

experiment = start(
  api_key="5svb9M3l7swM5UMvwTEEbJy9t",
  project_name="multi-angle-human-and-vehicle-tracking-system",
  workspace="ming12321"
)

# Report multiple hyperparameters using a dictionary:
hyper_params = {
   "learning_rate": 0.5,
   "steps": 100000,
   "batch_size": 50,
}
experiment.log_parameters(hyper_params)

# Initialize and train your model
# model = TheModelClass()
# train(model)

# Seamlessly log your Pytorch model
log_model(experiment, model=model, model_name="TheModel")

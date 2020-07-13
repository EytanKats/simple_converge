from tf_models.models_collection import models_collection

# Set parameters
model_args = dict()
model_args["model_name"] = "resnet_50"
model_args["classes_num"] = 3

model_args["regularizer_args"] = dict()
model_args["regularizer_args"]["regularizer_name"] = "l1_l2_regularizer"
model_args["regularizer_args"]["l1_reg_factor"] = 1e-3
model_args["regularizer_args"]["l2_reg_factor"] = 1e-2

# Create model
model_name = model_args["model_name"]
model_fn = models_collection[model_name]

model = model_fn()
model.parse_args(params=model_args)
model.build()

print(model.summary())

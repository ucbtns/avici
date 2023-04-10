import torch
from avici import model_torch

# Purpose: save weights of avici:
import pickle
import avici

# download pretrained model
model = avici.load_pretrained(download="scm-v0")


# state, loaded_config = load_checkpoint(root_path) --> from pretrained
# with open('model_data/state.pkl', 'wb') as file: pickle.dump(state, file)
# with open('model_data/configs.pkl', 'wb') as file: pickle.dump(loaded_config, file)
# with open('model_data/inference_model_kwargs.pkl', 'wb') as file: pickle.dump(inference_model_kwargs, file)
# with open('model_data/neural_net_kwargs.pkl', 'wb') as file: pickle.dump(neural_net_kwargs, file)

# Base Model
params = model.params # 167 params 
with open('model_data/base_model_parameters.pkl', 'wb') as file:
    pickle.dump(params, file)


# ================
# Load:
import pickle
import avici
from avici.utils.data_torch import torch_standardize_default_simple, torch_standardize_count_simple
from avici.pretrain import AVICIModel

with open('model_data/state.pkl', 'rb') as file: 
    state = pickle.load(file)

with open('model_data/configs.pkl', 'rb') as file: 
    configs = pickle.load(file)

with open('model_data/inference_model_kwargs.pkl', 'rb') as file: 
    inference_model_kwargs = pickle.load(file)

with open('model_data/neural_net_kwargs.pkl', 'rb') as file: 
    neural_net_kwargs = pickle.load(file)

import avici.model_torch as mt
inference_model = mt.InferenceModel(**inference_model_kwargs,model_class=mt.BaseModel, model_kwargs=neural_net_kwargs)

model = AVICIModel(params=state.params, model=inference_model, expects_counts=False,standardizer=torch_standardize_default_simple,)
# model._model.infer_edge_probs(model.params, torch.zeros([128,128]))




# ================
# Load:
import pickle
import avici
import torch
from avici import simulate_data
from avici.utils.data_torch import torch_standardize_default_simple, torch_standardize_count_simple
from avici.pretrain_torch import AVICIModel, load_pretrained
from avici.metrics import shd, classification_metrics, threshold_metrics

# Data
# g, x, _ = simulate_data(d=10, n=2000, domain="rff-gauss")

with open("model_data/data_x.pkl", 'rb') as file:
    x = pickle.load(file)

with open("model_data/data_g.pkl", 'rb') as file:
    g= pickle.load(file)

    
# Torch version:
x_tensor = torch.tensor(x, dtype=torch.float32)
modelTORCH = load_pretrained(download="scm-v0")
gg_prob = modelTORCH(x_tensor)

torch_prob = gg_prob.detach().numpy() 
avici.visualize(torch_prob, true=g, size=0.75)
print(f"SHD:   {shd(g, (torch_prob > 0.5).astype(int))}")
print(f"F1:    {classification_metrics(g, (torch_prob > 0.5).astype(int))['f1']:.4f}")
print(f"AUROC: {threshold_metrics(g, torch_prob)['auroc']:.4f}")

# modelTORCH._model.net.linear_layer_1.weight == torch.tensor(modelTORCH.params['BaseModel/linear']['w'].T)


# JAX version:
with open("model_data/data_x.pkl", 'wb') as file:
    pickle.dump(x, file)

from avici.pretrain import load_pretrained 
modelJAX = avici.load_pretrained(download="scm-v0")
g_prob = modelJAX(x=x)
avici.visualize(g_prob true=g, size=0.75)
print(f"SHD:   {shd(g, (g_prob > 0.5).astype(int))}")
print(f"F1:    {classification_metrics(g, (g_prob > 0.5).astype(int))['f1']:.4f}")
print(f"AUROC: {threshold_metrics(g, g_prob)['auroc']:.4f}")




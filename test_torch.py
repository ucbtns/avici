# Load:
import os
import pickle
import avici
import wandb
from avici import simulate_data
from avici.metrics import shd, classification_metrics, threshold_metrics
from avici.pretrain_torch import AVICIModel, load_pretrained as load_pretrained_torch
from avici.pretrain import load_pretrained as load_pretrained_jax
import itertools

os.environ['WANDB_API_KEY'] = '43d324138e70fc45bb22c0053b1586fc2e786fa4'
wandb.init(project='ABCI', entity='vilmarith', name='avici_baseline')

data_domain = ['lin-gauss', 'rff-gauss']
data_d = [10, 20, 50, 100]
data_n = [30, 100, 300, 1000] 

for domain in data_domain:
    for n in data_n:
        for d in data_d:
            for i in range(15):
                # Simulate data:
                g, x, _ = simulate_data(d=d, n=n, domain="lin-gauss") 

                # Torch version:
                from avici.pretrain_torch import AVICIModel, load_pretrained
                x_tensor = torch.tensor(x, dtype=torch.float32)
                modelTORCH = load_pretrained(download="scm-v0")
                gg_prob = modelTORCH(x_tensor)
                torch_prob = gg_prob.detach().numpy() 
                avici.visualize(torch_prob, true=g, size=0.75)
                wandb.log({"torch_SHD": shd(g, (torch_prob > 0.5).astype(int)), 
                        "torch_F1": classification_metrics(g, (torch_prob > 0.5).astype(int))['f1'],
                        "torch_AUROC": threshold_metrics(g, torch_prob)['auroc']})

                # JAX version:
                from avici.pretrain import load_pretrained 
                modelJAX = avici.load_pretrained(download="scm-v0")
                g_prob = modelJAX(x=x)
                avici.visualize(g_prob, true=g, size=0.75)
                wandb.log({"JAX_SHD": shd(g, (torch_prob > 0.5).astype(int)), 
                        "JAX_F1": classification_metrics(g, (torch_prob > 0.5).astype(int))['f1'],
                        "JAX_AUROC": threshold_metrics(g, torch_prob)['auroc']})



# Combine loop variables into a single iterable using itertools.product
for domain, n, d, i in itertools.product(data_domain, data_n, data_d, range(15)):
    # Simulate data:
    g, x, _ = simulate_data(d=d, n=n, domain=domain) 
    name = 'comparison'+str(d)+str(n)+domain

    # Torch version:
    modelTORCH = load_pretrained_torch(download="scm-v0")
    torch_prob = modelTORCH(torch.tensor(x, dtype=torch.float32)).detach().numpy() 
    avici.visualize(torch_prob, true=g, size=0.75, True, 'torch_'+name)

    # JAX version:
    modelJAX = load_pretrained_jax(download="scm-v0")
    jax_prob = modelJAX(x=x)
    avici.visualize(jax_prob, true=g, size=0.75, True, 'JAX_'+name)

    # Calculate metrics
    binary_tprob = (torch_prob > 0.5).astype(int)
    torch_shd = shd(g, binary_tprob)
    torch_f1 = classification_metrics(g, binary_tprob)['f1']
    torch_auroc = threshold_metrics(g, torch_prob)['auroc']

    binary_jprob = (jax_prob > 0.5).astype(int)
    jax_shd = shd(g, binary_jprob)
    jax_f1 = classification_metrics(g, binary_jprob)['f1']
    jax_auroc = threshold_metrics(g, jax_prob)['auroc']

    # Log metrics
    wandb.log({
        'd':d, 'n':n, 'i':i, 'domain':domain
        "torch_SHD": torch_shd,
        "torch_F1": torch_f1,
        "torch_AUROC": torch_auroc,
        "JAX_SHD": jax_shd,
        "JAX_F1": jax_f1,
        "JAX_AUROC": jax_auroc
    })

wandb.finish()


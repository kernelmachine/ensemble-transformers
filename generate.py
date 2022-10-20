from ensemble_transformers import EnsembleModelForCausalLM
from transformers import AutoTokenizer
import torch
import os
import subprocess
import pickle
import numpy as np
import random

def initialize_slurm_distributed(num_gpus=8, master_addr="127.0.0.1", master_port=29500):
    rank = int(os.environ['SLURM_PROCID'])
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ['RANK'] = os.environ['SLURM_NODEID']
    os.environ['WORLD_SIZE'] = str(num_gpus)
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(f'{i}' for i in range(num_gpus))

    node_list = os.environ.get("SLURM_JOB_NODELIST")
    hostnames = subprocess.check_output(
        ["scontrol", "show", "hostnames", node_list]
    )
    init_method= "tcp://{host}:{port}".format(
        host=hostnames.split()[0].decode("utf-8"),
        port=os.environ['MASTER_PORT'],
    )
    torch.distributed.init_process_group(backend="nccl", init_method=init_method, world_size=num_gpus, rank=rank)
    print("initialized!")

def load_model(path_to_model):
    with open(path_to_model, 'rb') as f:
        out = pickle.load(f)
    return out
        
    
def main(cluster, context="", priors=None, num_clusters=2, seed=0):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    models = {i : f'/private/home/suching/cluster/models/EXPERIMENT=mod_MODEL=facebook-opt-125m_GPUS=16_NODES=2_TAG=cbtm_NUM_CLUSTERS={num_clusters}_CLUSTER={i}/checkpoint-10000/' for i in range(num_clusters)}
    tokenizer = AutoTokenizer.from_pretrained(models[torch.distributed.get_rank()], use_fast=False)
    ensemble = EnsembleModelForCausalLM.from_multiple_pretrained(models[torch.distributed.get_rank()])
    ensemble.to_multiple([f"cuda:{torch.distributed.get_rank()}"])
    
    processor_kwargs = {"add_special_tokens": True, "truncation": True, "max_length": 1024, "return_tensors": 'pt'}
    batch = [context]

    inputs = tokenizer(batch, **processor_kwargs)
    
    if cluster:
        vectorizer = load_model("/private/home/suching/demix-data/c4_domain_clusters/tfidf.pkl")
        kmeans = load_model("/private/home/suching/demix-data/c4_domain_clusters/kmeans.pkl")
    else:
        vectorizer = None
        kmeans = None
    if vectorizer:
        from transformers import TopPLogitsWarper
    else:
        from ensemble_transformers import TopPLogitsWarper
    nucleus = TopPLogitsWarper(top_p=0.9)
    priors = priors
    if cluster:
        kwargs = dict(kmeans=kmeans, vectorizer=vectorizer,tokenizer=tokenizer)
    else:
        kwargs = dict(priors=priors)
    outputs = ensemble.sample(inputs.input_ids,
                            logits_warper=nucleus,
                            max_length=100,
                            **kwargs)
    if torch.distributed.get_rank() == 0:
        print(tokenizer.batch_decode(outputs, skip_special_tokens=True))

if __name__ == '__main__':
   # initialize_slurm_distributed(2, master_port=29500)
    # priors = [1/8] * 8
    # COVID19-TWEETS
 #   priors =[0.04015719, 0.02896253, 0.26695506, 0.27616504, 0.01892541, 0.17184799,  0.09064946, 0.10633733]
  
    init_method= "tcp://{host}:{port}".format(
                    host=os.environ['MASTER_ADDR'],
                            port=os.environ['MASTER_PORT'],
                                )
    torch.distributed.init_process_group(backend="nccl", init_method=init_method, world_size=2, rank=int(os.environ['SLURM_PROCID']))
    main(cluster=False, context="@", priors=[1.0, 0.0], seed=4, num_clusters=2)
    

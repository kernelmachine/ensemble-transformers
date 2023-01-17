from ensemble_transformers import EnsembleModelForCausalLM
from transformers import AutoTokenizer
import torch
import os
import subprocess
import pickle
import numpy as np
import random



def initialize_slurm_distributed(num_gpus=8, master_addr="127.0.0.1", master_port=29500):
    print("initializing slurm...")
    rank = int(os.environ['SLURM_PROCID'])
    node_list = os.environ.get("SLURM_JOB_NODELIST")
    hostnames = subprocess.check_output(
        ["scontrol", "show", "hostnames", node_list]
    )
    master_addr = hostnames.split()[0].decode("utf-8")
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ['RANK'] = os.environ['SLURM_NODEID']
    os.environ['WORLD_SIZE'] = str(num_gpus)
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(f'{i}' for i in range(num_gpus))

    init_method= "tcp://{host}:{port}".format(
        host=os.environ['MASTER_ADDR'],
        port=str(master_port),
    )
    print("initializing at ", init_method)
    torch.distributed.init_process_group(backend="nccl", init_method=init_method, world_size=num_gpus, rank=rank)
    torch.distributed.barrier()
    print("initialized!")

def load_model(path_to_model):
    with open(path_to_model, 'rb') as f:
        out = pickle.load(f)
    return out
        
    
def main(cluster, context="", priors=None, num_clusters=2, seed=0):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    models = {i: f'/checkpoint/suching/hf_models/mod/finetune.opt.opt_data.1.3b.numclusters2.cluster{i}.0edr.mu10000.wu0.bsz8.fp16adam.rs1234.lr2e-05.pat_10.ngpu16' for i in range(num_clusters)}
    print(models)
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m", use_fast=False)
    ensemble = EnsembleModelForCausalLM.from_multiple_pretrained(models[torch.distributed.get_rank()])
    ensemble.to_multiple([f"cuda:{torch.distributed.get_rank()}"])
    processor_kwargs = {"add_special_tokens": True, "truncation": True, "max_length": 2048, "return_tensors": 'pt'}
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
    nucleus = TopPLogitsWarper(top_p=0.95)
    priors = priors
    if cluster:
        kwargs = dict(kmeans=kmeans, vectorizer=vectorizer,tokenizer=tokenizer)
    else:
        kwargs = dict(priors=priors)
    print("generating output...")
    outputs = ensemble.sample(inputs.input_ids,
                            logits_warper=nucleus,
                            max_length=100,
                            **kwargs)
    if torch.distributed.get_rank() == 0:
        print(tokenizer.batch_decode(outputs, skip_special_tokens=True))

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def init_distributed_mode(world_size):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        rank = int(os.environ['SLURM_PROCID'])
        gpu = rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        distributed = False
        return

    distributed = True

    torch.cuda.set_device(gpu)
    dist_backend = 'nccl'
    dist_url= "tcp://{host}:{port}".format(
        host=os.environ['MASTER_ADDR'],
        port=os.environ['MASTER_PORT'],
    )
    print('| distributed init (rank {}): {}'.format(
        rank, dist_url), flush=True)
    
    torch.distributed.init_process_group(backend="nccl", init_method=dist_url,
                                         world_size=world_size, rank=rank)
    torch.distributed.barrier()
    # setup_for_distributed(rank == 0)


if __name__ == '__main__':
    init_distributed_mode(2)
    # priors = [1/8] * 8
    # COVID19-TWEETS
 #   priors =[0.04015719, 0.02896253, 0.26695506, 0.27616504, 0.01892541, 0.17184799,  0.09064946, 0.10633733]
  
    # init_method= "tcp://{host}:{port}".format(
                    # host=os.environ['MASTER_ADDR'],
                            # port=os.environ['MASTER_PORT'],
                                # )
    # torch.distributed.init_process_group(backend="nccl", init_method=init_method, rank=int(os.environ['SLURM_PROCID']))
    # print(torch.distributed.get_rank())
    main(cluster=False, context="My name is", priors=[0.0, 1.0], seed=np.random.randint(0,2500), num_clusters=2)
    

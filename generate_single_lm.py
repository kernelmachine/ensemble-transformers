from transformers import AutoTokenizer, AutoModelForCausalLM, TopPLogitsWarper
import torch
import os
import subprocess
import pickle




def load_model(path_to_model):
    with open(path_to_model, 'rb') as f:
        out = pickle.load(f)
    return out
    

if __name__ == '__main__':
    
    # models = {i : f'/private/home/suching/cluster/models/EXPERIMENT=mod_MODEL=facebook-opt-125m_GPUS=2_NODES=1_CLUSTER={i}/checkpoint-10000/' for i in range(8)}
    import numpy as np
    
    import random
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    models = {}
    models[0] = "facebook/opt-125m"
    models[1] = "/private/home/suching/cluster/models/EXPERIMENT=dense_MODEL=facebook-opt-125m_GPUS=16_NODES=2/checkpoint-10000/"
    tokenizer = AutoTokenizer.from_pretrained(models[0], use_fast=False)
    for key in models:
        model = AutoModelForCausalLM.from_pretrained(models[key])
        model = model.to("cuda:1")

        processor_kwargs = {"add_special_tokens": True, "truncation": True, "max_length": 1024, "return_tensors": 'pt'}
        batch = ["#blessed @"]
        inputs = tokenizer(batch, **processor_kwargs)
        # nucleus = TopPLogitsWarper(top_p=0.9)
        outputs = model.sample(inputs.input_ids.to(model.device),
                                # logits_warper=nucleus,
                                max_length=100)
        print(f"Model {key}: {tokenizer.batch_decode(outputs, skip_special_tokens=True)}")
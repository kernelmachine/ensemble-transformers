from ensemble_transformers import EnsembleModelForCausalLM, TopPLogitsWarper
import torch
from transformers import StoppingCriteriaList, MaxLengthCriteria, AutoTokenizer
import itertools
from accelerate import Accelerator
from datasets import load_dataset
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import os
import subprocess
import random
import pickle
import argparse

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

def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def gather_target_probs(probs, target):
    probs = probs.gather(
        dim=2,
        index=target.unsqueeze(-1),
    )
    return probs



def main(cluster, file, seed=42, precomputed_prior=None):

    set_seed(seed)

    accelerator = Accelerator()


    accelerator_log_kwargs = {}

    tokenizer = AutoTokenizer.from_pretrained('facebook/opt-125m')

    train_file = None
    validation_file = file

    data_files = {}
    dataset_args = {}
    if train_file is not None:
        data_files["train"] = train_file
    if validation_file is not None:
        data_files["validation"] = validation_file
    extension = (
        train_file.split(".")[-1]
        if train_file is not None
        else validation_file.split(".")[-1]
    )
    if extension == "txt":
        extension = "text"
        dataset_args["keep_linebreaks"] = True
    raw_datasets = load_dataset(extension, data_files=data_files, cache_dir=None, use_auth_token=None, **dataset_args)
    
    # # If no validation data is there, validation_split_percentage will be used to divide the dataset.
    # if "validation" not in raw_datasets.keys():
    #     raw_datasets["validation"] = load_dataset(
    #         extension,
    #         data_files=data_files,
    #         split=f"train[:{data_args.validation_split_percentage}%]",
    #         cache_dir=model_args.cache_dir,
    #         use_auth_token=True if model_args.use_auth_token else None,
    #         **dataset_args,
    #     )
    #     raw_datasets["train"] = load_dataset(
    #         extension,
    #         data_files=data_files,
    #         split=f"train[{data_args.validation_split_percentage}%:]",
    #         cache_dir=model_args.cache_dir,
    #         use_auth_token=True if model_args.use_auth_token else None,
    #         **dataset_args,
    #     )


    def tokenize_function(examples):
        return tokenizer(examples['text'])

    with accelerator.main_process_first():
        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            num_proc=1,
            remove_columns=["text"],
            load_from_cache_file=False,
            desc="Running tokenizer on dataset",
        )




    block_size = 1024
    if block_size is None:
            block_size = tokenizer.model_max_length
            if block_size > 1024:
                logger.warning(
                    f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                    "Picking 1024 instead. You can change that default value by passing --block_size xxx."
                )
            block_size = 1024
    else:
        if block_size > tokenizer.model_max_length:
            logger.warning(
                f"The block_size passed ({block_size}) is larger than the maximum length for the model"
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(block_size, tokenizer.model_max_length)

    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(itertools.chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    with accelerator.main_process_first():
            lm_datasets = tokenized_datasets.map(
                group_texts,
                batched=True,
                num_proc=1,
                load_from_cache_file=False,
                desc=f"Grouping texts in chunks of {block_size}",
            )

    def pairwise_distance(data1, data2, device=torch.device('cpu'), tqdm_flag=True):
        # if tqdm_flag:
            # print(f'device is :{device}')
        
        # transfer to device
        data1, data2 = data1.to(device), data2.to(device)

        # N*1*M
        A = data1.unsqueeze(dim=1)

        # 1*N*M
        B = data2.unsqueeze(dim=0)

        dis = (A - B) ** 2.0
        # return N*N matrix for pairwise distance
        dis = dis.sum(dim=-1).squeeze()
        return dis
    def generate_context_clusters(examples, tokenizer, vectorizer, kmeans):        
        # c_ = clusters[id]
        clusters = []
        for i in range(50):
            decoded_text = [tokenizer.decode(x[:i]) for x in examples['input_ids']]
            vectorized_text = vectorizer.transform(decoded_text)
            c_, dists =  kmeans.predict(torch.from_numpy(vectorized_text), return_distances=True)
            probs = torch.nn.functional.softmax(dists, dim=-1)
            clusters.append(c_.unsqueeze(1))
        examples['context_clusters'] = torch.cat(clusters, 1).numpy()
        # c_ = [cached_clusters.get(domain[i], backup_clusters[i]) for i in range(len(domain))]
        # c_ =  kmeans.predict(torch.from_numpy(vectorizer.transform(text))).cpu().numpy()
        
        # from fairseq import pdb; pdb.set_trace()
        # res = [z for z,y in zip(batch['text'], c_) if y == cluster]
        return examples


    if cluster:
        with accelerator.main_process_first():
            def load_model(path_to_model):
                with open(path_to_model, 'rb') as f:
                    out = pickle.load(f)
                return out
            vectorizer = load_model("/private/home/suching/demix-data/c4_domain_clusters/tfidf.pkl")
            kmeans = load_model("/private/home/suching/demix-data/c4_domain_clusters/kmeans.pkl")
            lm_datasets = lm_datasets.map(
                lambda x: generate_context_clusters(x, tokenizer, vectorizer, kmeans),
                batched=True,
                num_proc=1,
                load_from_cache_file=False,
                desc="Generating context clusters",
            )



    # # select examples that are in a single cluster
    # with training_args.main_process_first(desc="dataset map clustering"):
        
        
    #     cached_clusters = load_model(data_args.path_to_cached_clusters)
    #     if training_args.do_train:
    #         raw_datasets['train'] = tokenized_datasets['train'].filter(lambda example: example_in_cluster(example['text'], example['domain'], data_args.train_cluster, vectorizer, kmeans, cached_clusters), batched=True)
    #     if training_args.do_eval:
    #         raw_datasets['validation'] = raw_datasets['validation'].filter(lambda example: example_in_cluster(example['text'], example['domain'], data_args.train_cluster, vectorizer, kmeans, cached_clusters), batched=True)
            
    # train_dataset = lm_datasets["train"]
    eval_dataset = lm_datasets["validation"]
    models = {i : f'/private/home/suching/cluster/models/EXPERIMENT=mod_MODEL=facebook-opt-125m_GPUS=2_NODES=1_CLUSTER={i}/checkpoint-10000/' for i in range(8)}

    # modes = {0: "gpt2", 1: "lvwerra/gpt2-imdb"}
    torch.distributed.barrier()



    ensemble = EnsembleModelForCausalLM.from_multiple_pretrained(models[torch.distributed.get_rank()])


        
    # 
    for i in range(len(ensemble.models)):
        ensemble.models[i].eval()
    #     ensemble.models[i] = accelerator.prepare(ensemble.models[i])
        
    # import pdb; pdb.set_trace()
    # batch = ["<|endoftext|>"]
    # processor_kwargs = {"add_special_tokens": True, "truncation": True, "max_length": 1024, "return_tensors": 'pt'}
    # inputs = ensemble.preprocessors[0](batch, **processor_kwargs)
    # nucleus = TopPLogitsWarper(top_p=0.9)
    # priors  = [1.0, 0.0]
    # outputs = ensemble.sample(inputs.input_ids, max_length=100, logits_warper=nucleus, priors=priors)
    # print(ensemble.preprocessors[0].batch_decode(outputs, skip_special_tokens=True))

    from transformers import default_data_collator
    batch_size = 2
    eval_dataloader = DataLoader(
            eval_dataset, collate_fn=default_data_collator, batch_size=batch_size
    )

    # model1, model2, eval_dataloader = accelerator.prepare(*ensemble.models, eval_dataloader)

    ensemble.to_multiple([f"cuda:{torch.distributed.get_rank()}"])


    expert_probs_all = []
    world_size = torch.distributed.get_world_size()
    priors = [1/ world_size] * world_size
    losses = []
    pbar = tqdm(eval_dataloader, disable=torch.distributed.get_rank() != 0)
    # precomputed_prior = [5.96925271e-01, 2.29693412e-01, 4.38744091e-04,  6.46805285e-02, 2.19283846e-03, 5.31826369e-03, 9.66890400e-02, 4.06191127e-03]
    # precomputed_prior=[0.48553511, 0.08346399, 0.00672293, 0.00311078, 0.00409492, 0.04728625, 0.20530888, 0.16447713]

    # precomputed_prior = [4.06190318e-01, 2.12151453e-01,  3.40328294e-04, 3.82875938e-03, 1.10094062e-02, 1.98403994e-01, 1.55755237e-01, 1.23205058e-02]
    # precomputed_prior = None
    # precomputed_prior = [1, 0]
    with torch.no_grad():
        counter =0
        for batch in pbar:
            #counter+=1 
            #if counter==100:
            #    break
            outputs = ensemble(input_ids=batch['input_ids'],
                                attention_mask=batch['attention_mask'],
                                labels=batch['labels'],
                                context_clusters=batch['context_clusters'] if cluster else None,
                                priors=priors)
            if not cluster:
                # exp_probs = []
                # for i in range(batch.shape[1]):
                #     exp_probs.append(outputs['expert_probs'][:,i,batch['labels'][i,:]].unsqueeze(i))
                # expert_probs = torch.cat(exp_probs, dim=1)
                expert_probs = outputs['expert_probs'].mean(1).unsqueeze(0).cpu().numpy()
                # expert_probs = outputs['expert_probs'][:,:,batch['labels'][:,-1].squeeze(0)].mean(1).mean(1).unsqueeze(0).cpu().detach().numpy()
                expert_probs_all.append(expert_probs)
                priors = pd.DataFrame(np.concatenate(expert_probs_all,0)).ewm(alpha=0.3, adjust=False).mean().tail(n=1).to_numpy().squeeze(0)
        # if not precomputed_prior:
            
            
            losses.append(outputs['loss'].item())
            pbar.set_description(f"ppl: {np.exp(np.mean(losses))}, priors: {precomputed_prior or priors}")
        if torch.distributed.get_rank() == 0:   
            print(f"final ppl: {np.exp(np.mean(losses))}")
            print(priors)


if __name__ == '__main__':
    initialize_slurm_distributed(8, master_port=29500)

    parser = argparse.ArgumentParser()
    parser.add_argument("--file")
    parser.add_argument("--precomputed_prior", type=int, nargs="+")
    args = parser.parse_args()
    # file = "../raw_data/demix_scale/covidtweets/splits/dev.txt"
    # file = "/private/home/suching/demix-data/cluster_1.txt"

    # PPL 37.204686391350926
    #[2.42843189e-09, 3.73016554e-01, 4.09775106e-01, 5.01054346e-030,  1.83860220e-06, 2.11473275e-01, 6.98363135e-04, 2.43094799e-05]

    #precomputed_prior = [0.8, 0.0, 0.1,0.0,0,0,0.1,0]
#     precomputed_prior = [1.53432509e-06 3.32477824e-01 4.03767907e-01 2.28626977e-02 
# 0:  1.14986346e-03 2.31018428e-01 7.26075859e-03 1.46098221e-03]:
    #precomputed_prior =  [6.00246925e-01, 3.04175785e-02, 1.23715056e-01, 8.35959456e-02, 3.98626546e-03, 1.17209272e-02, 1.46314733e-01, 2.58234944e-06]
    #precomputed_prior = [0.02394467, 0.87081061, 0.02492725, 0.01586126, 0.00966152, 0.02538445, 0.01510562, 0.01430462]
    # precomputed_prior = [1.0, 0,0,0,0,0,0,0]
    main(cluster=True, 
        file=args.file,
        seed=24,
        precomputed_prior=args.precomputed_prior)

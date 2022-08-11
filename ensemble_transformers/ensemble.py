from typing import List, Union

import numpy as np
import torch
from ensemble_transformers.base import EnsembleBaseModel
from torch.nn import functional as F
from overrides import overrides
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
import warnings
from transformers.generation_logits_process import (
    LogitsWarper,
    LogitsProcessorList
)
from transformers.generation_stopping_criteria import (
    MaxLengthCriteria,
    MaxTimeCriteria,
    StoppingCriteria,
    StoppingCriteriaList,
    validate_stopping_criteria,
)
from transformers.pytorch_utils import torch_int_div
from transformers.generation_utils import SampleOutput
from transformers.utils import ModelOutput, logging



class TopPLogitsWarper(LogitsWarper):
    """
    [`LogitsWarper`] that performs top-p, i.e. restricting to top tokens summing to prob_cut_off <= prob_cut_off.
    Args:
        top_p (`float`):
            If set to < 1, only the most probable tokens with probabilities that add up to `top_p` or higher are kept
            for generation.
        filter_value (`float`, *optional*, defaults to `-float("Inf")`):
            All filtered values will be set to this float value.
        min_tokens_to_keep (`int`, *optional*, defaults to 1):
            Minimum number of tokens that cannot be filtered.
    """

    def __init__(self, top_p: float, filter_value: float = -float("Inf"), min_tokens_to_keep: int = 1):
        top_p = float(top_p)
        if top_p < 0 or top_p > 1.0:
            raise ValueError(f"`top_p` has to be a float > 0 and < 1, but is {top_p}")

        self.top_p = top_p
        self.filter_value = filter_value
        self.min_tokens_to_keep = min_tokens_to_keep

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        sorted_logits, sorted_indices = torch.sort(scores, descending=True)
        cumulative_probs = sorted_logits.cumsum(dim=-1)

        # Remove tokens with cumulative top_p above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > self.top_p
        if self.min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., : self.min_tokens_to_keep - 1] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        scores = scores.masked_fill(indices_to_remove, self.filter_value)
        return scores



class EnsembleModelForCausalLM(EnsembleBaseModel):


    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        main_device: Union[str, torch.device] = "cpu",
        priors: Optional[List[float]] = None,

        preprocessor_kwargs: dict = {"add_special_tokens": True, "truncation": True, "max_length": 1024, "return_tensors": 'pt'},
    ):
        outputs = []
        for model in self.models:
            output = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = output['logits'].softmax(-1)
            outputs.append(probs.unsqueeze(0))
        if not priors:
            priors = [1 / len(self.models)] * len(self.models)
        model_probs = torch.cat(outputs, dim=0)
        weights = model_probs[:, :, :-1].clone()

        denom = weights.clone()

        for ix, prior in enumerate(priors):
            denom[ix, :].mul_(prior)

        denom = denom.sum(0)

        for ix, prior in enumerate(priors):
            weights[ix, :].mul_(prior).div_(denom)

        print(f"priors: {priors}")
        beginning_weights = torch.tensor(priors).float().repeat(model_probs.shape[1], 1).t().unsqueeze(-1).repeat(1,1,1,model_probs.shape[-1]).transpose(0,1).to(weights)

        weights = torch.cat([beginning_weights, weights], 2)
        probs = torch.einsum("ebsv,ebsv->bsv", (weights,model_probs))
        probs.log_()

        if labels:
            loss = torch.nn.NLLLoss(reduction='mean')
            shift_probs = probs[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            l  = loss(shift_probs.view(-1, 50257), shift_labels.view(-1))

            return {"loss": l, "probs": probs}
        else:
            return {"probs": probs}
        # if return_all_outputs:
        # return 

        
        # return torch.stack(
        #     [weight * output.logits.to(main_device) for weight, output in zip(self.config.weights, outputs)]
        # ).sum(dim=0)
    @overrides
    def sample(
        self,
        input_ids: torch.LongTensor,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        logits_warper: Optional[LogitsProcessorList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        synced_gpus: Optional[bool] = False,
        priors: Optional[List[float]] = None,

        **model_kwargs) -> Union[SampleOutput, torch.LongTensor]:
        # init values
        input_ids = torch.LongTensor(input_ids)
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if max_length is not None:
            # warnings.warn(
            #     "`max_length` is deprecated in this function, use"
            #     " `stopping_criteria=StoppingCriteriaList(MaxLengthCriteria(max_length=max_length))` instead.",
            #     UserWarning,
            # )
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
        logits_warper = logits_warper if logits_warper is not None else LogitsProcessorList()
        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id
        output_scores = output_scores if output_scores is not None else self.config.output_scores
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate if return_dict_in_generate is not None else self.config.return_dict_in_generate
        )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        # keep track of which sequences are already finished
        unfinished_sequences = input_ids.new(input_ids.shape[0]).fill_(1)
        cur_len = input_ids.shape[-1]

        this_peer_finished = False  # used by synced_gpus only
        # auto-regressive generation
        i = 0
        while True:
            i+= 1
            print(i)
            if synced_gpus:
                # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
                # The following logic allows an early break if all peers finished generating their sequence
                this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
                # send 0.0 if we finished, 1.0 otherwise
                dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                # did all peers finish? the reduced sum will be 0.0 then
                if this_peer_finished_flag.item() == 0.0:
                    break
            
            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            # forward pass to get next token
            outputs = self(
                **model_inputs,
                priors=priors,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

            if synced_gpus and this_peer_finished:
                cur_len = cur_len + 1
                continue  # don't waste resources running the code we don't need

            next_token_scores = outputs['probs'][:, -1, :]

            # pre-process distribution
            next_token_scores = logits_processor(input_ids, next_token_scores)
            next_token_scores = logits_warper(input_ids, next_token_scores)

            # Store scores, attentions and hidden_states when required
            # if return_dict_in_generate:
            #     if output_scores:
            #         scores += (next_token_scores,)
            #     if output_attentions:
            #         decoder_attentions += (
            #             (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
            #         )
            #         if self.config.is_encoder_decoder:
            #             cross_attentions += (outputs.cross_attentions,)

            #     if output_hidden_states:
            #         decoder_hidden_states += (
            #             (outputs.decoder_hidden_states,)
            #             if self.config.is_encoder_decoder
            #             else (outputs.hidden_states,)
            #         )

            # sample
            # probs = nn.functional.softmax(next_token_scores, dim=-1)
            next_tokens = torch.multinomial(torch.exp(next_token_scores.view(-1, 50257)), num_samples=1).squeeze(1)
            # import pdb; pdb.set_trace()
            # finished sentences should have their next token be a padding token
            if eos_token_id is not None:
                if pad_token_id is None:
                    raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )
            cur_len = cur_len + 1

            # if eos_token was found in one sentence, set sentence to finished
            if eos_token_id is not None:
                unfinished_sequences = unfinished_sequences.mul((next_tokens != eos_token_id).long())

            # stop when each sentence is finished, or if we exceed the maximum length
            if unfinished_sequences.max() == 0 or stopping_criteria(input_ids, scores):
                if not synced_gpus:
                    break
                else:
                    this_peer_finished = True

        # if return_dict_in_generate:
        #     if self.config.is_encoder_decoder:
        #         return SampleEncoderDecoderOutput(
        #             sequences=input_ids,
        #             scores=scores,
        #             encoder_attentions=encoder_attentions,
        #             encoder_hidden_states=encoder_hidden_states,
        #             decoder_attentions=decoder_attentions,
        #             cross_attentions=cross_attentions,
        #             decoder_hidden_states=decoder_hidden_states,
        #         )
        #     else:
        #         return SampleDecoderOnlyOutput(
        #             sequences=input_ids,
        #             scores=scores,
        #             attentions=decoder_attentions,
        #             hidden_states=decoder_hidden_states,
        #         )
        # else:
        return input_ids
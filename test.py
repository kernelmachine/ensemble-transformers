from ensemble_transformers import EnsembleModelForCausalLM, TopPLogitsWarper
import torch
from transformers import StoppingCriteriaList, MaxLengthCriteria


ensemble = EnsembleModelForCausalLM.from_multiple_pretrained("lvwerra/gpt2-imdb", "neulab/gpt2-finetuned-wikitext103")

# batch = ["""
# <|endoftext|> The only reason I wanted to see this was because of Orlando Bloom. Simply put, the movie was spectacularly average. It's not bad, but it's really not very good. The editing is good; the film is well-paced. The direction is competent and assured. The story is plodding. The film is averagely acted by Ledger, Bloom, and the normally great Watts and Rush. The accents are impenetrable if you're from the US so just sit back and enjoy the scenery (or as I like to call it, Orlando Bloom). By the end of the film, I was neither bored nor moved. Some people have asked what happened to Ned Kelly at the end of the movie. I have to say, I so did not care by that point.Really, the only reason I can recommend this is that Orlando Bloom kind of, sort of shows some hints of range (although the oft-present "I'm pretty and confused" look is prominent), so fangirls may find it worth the matinee price. Other than that, just don't see it. It's neither good enough nor bad enough to be entertaining.
# <|endoftext|> I remember watching "Lost Missile" (actually throwing a fit until my brother and several cousins at whose home I was an overnight guest agreed to watch it with me - I was, from time to time, the Eric Cartman of the 1960s - sorry, guys) and being somewhat embarrassed when the sustained wave of million-degree heat emerged as a plot device - even as a second-grader I knew that a mere missile just couldn't carry the energy around for that much heat or devastation over more than the duration and limited radius of a nuclear detonation. My inflicting that turkey on loving relatives was a self-punishing crime.The film's production values were very good. The acting isn't bad (apart from the Shatnerism of the actor who played governor's aide that someone else here mentioned).But the idea of a missile Easy-Baking the surface of the Earth by means of the heat of its exhaust... no.How'd the people at "Mystery Science Theater 3000" miss "The Lost Missile," anyway? It's a great classic of unintentional comedy - watch it if you want something to drink beer to some weekend.
# <|endoftext|> This 1997 film-blanc classic tale of smoldering passion has achieved its well-deserved legendary status
# """
# ]

batch = ["<|endoftext|>"]
processor_kwargs = {"add_special_tokens": True, "truncation": True, "max_length": 1024, "return_tensors": 'pt'}
inputs = ensemble.preprocessors[0](batch, **processor_kwargs)
# stopping_criteria = StoppingCriteriaList([MaxLengthCriteria(max_length=1050)])
nucleus = TopPLogitsWarper(top_p=0.9)
priors  = [0.05, 0.95]
outputs = ensemble.sample(inputs.input_ids, max_length=100, logits_warper=nucleus, priors=priors)

print(ensemble.preprocessors[0].batch_decode(outputs, skip_special_tokens=True))

# loss, probs, weights, loss_from_each_model = ensemble(batch)
# print(loss)
# # print(weights)
# print([torch.exp(x) for x in loss_from_each_model)
# torch.nn.NLLLoss(model_probs, reduction='mean')


# weights = model_probs[:, :, :-1].clone()

# priors = [0.5, 0.5]

# denom = weights.clone()

# for ix, prior in enumerate(priors):
#     denom[ix, :].mul_(prior)


# denom = denom.sum(0)

# for ix, prior in enumerate(priors):
#     weights[ix, :].mul_(prior).div_(denom)


# beginning_weights = torch.tensor(priors).float().repeat(model_probs.shape[1], 1).t().unsqueeze(-1).repeat(1,1,1,model_probs.shape[-1]).transpose(0,1).to(weights)


# weights = torch.cat([beginning_weights, weights], 2)

# avg_probs = torch.einsum("ebsv,ebsv->bsv", (weights,model_probs))

# # avg_probs.log_()

# import math
# print(2 ** (-avg_probs.sum() / avg_probs.numel() / math.log(2)))
# LLM-fine-tuning
## Motivation
Full fine-tuning of Large Language Models is the process of turning general-purpose model into specialized one while updating all pre-trained model weights over a downstream task dataset. Full fine-tuning is exceedingly resource-intensive, that is why various fine-tuning acceleration strategies are utilized, allowing for reduction of training time and memory costs, while maintaining the quality. There are exist different approaches to fine-tuning acceleration. Among them, Parameter-Efficient Fine-Tuning (PEFT) considered as the most prominent approach, which updates only a small fraction of the model parameters, substantially reducing the computational and storage costs while maintaining the accuracy. At the same time, there is a large number of works in the field of detection the super weights and activations inside LLM, that demonstrates considerable acceleration and memory reduction withou sacrificing the accuracy.

## Idea
This work aims to accelerate fine-tuning of LLM with effective combination of PEFT and identification of super weights and activations inside the model. 

## Plan
- [ ] Choose models.
- [ ] Choose datasets.
- [ ] Identify super weights and activations inside the model.
- [ ] Propose efficient fine-tuning approach.
- [ ] Benchmark proposed fine-tuning approach on chosen models and datasets and compare with existing SOTA approaches.

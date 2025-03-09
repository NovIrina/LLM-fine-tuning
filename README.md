# LLM-fine-tuning
## Motivation
Full fine-tuning of Large Language Models is the process of turning general-purpose model into specialized one while updating all pre-trained model weights over a downstream task dataset. Full fine-tuning is exceedingly resource-intensive, that is why various fine-tuning acceleration strategies are utilized, allowing for reduction of training time and memory costs, while maintaining the quality. There are exist different approaches to fine-tuning acceleration. Among them, Parameter-Efficient Fine-Tuning (PEFT) considered as the most prominent approach, which updates only a small fraction of the model parameters, substantially reducing the computational and storage costs while maintaining the accuracy. At the same time, there are works postulating the existence of a small fraction of extremely important [Super Weights inside LLM](https://arxiv.org/abs/2411.07191), removing of which can lead to substantial quality degradation. This project aims at training only layers containing these Super Weights to further accelerate LoRA-based fine-tuning while maintaining the quality.

## Prerequisites
- Model: [GPT2](https://huggingface.co/openai-community/gpt2);
- Dataset: [E2E-NLG](https://huggingface.co/datasets/tuetschek/e2e_nlg) - dataset for training end-to-end, data-driven natural language generation systems in the restaurant domain;
- Baseline: [LoRA](https://arxiv.org/abs/2106.09685)-based fine-tuning;

## Setup
1. Create virtual environment
```bash
    python3.11 -m venv venv
    source venv/bin/activate
```
2. Install project dependencies
```bash
    pip install -r requirements.txt
```

## Run training 

## Run evaluation

## Run Super Weights identification 

### Arguments 
* `--path_to_model` - path to your model, by default: `openai-community/gpt2`
* `--path_to_tokenizer` - path to model tokenizer, by default: `openai-community/gpt2`

### Run
```python
python -m super_weights_identification.main
```


## Run web application 

## Results 


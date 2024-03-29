# Transformer sentence embeddings

Simple and lean implementation for computing sentence embeddings for different transformer-based models. BERT-flow implementation based on the code from [repository](https://github.com/UKPLab/pytorch-bertflow). Contrastive tension implementation based on the implementation of [Sentence Transformers](https://github.com/UKPLab/sentence-transformers).

Among other things changes comprise:
* Added monitoring ([Weights and Biases](https://wandb.ai), which of course can be replaced by any other monitoring service)
* Added evaluation ([SentEval](https://github.com/facebookresearch/SentEval)) script

#### News
- **08/19/2022:** :confetti_ball: Added Enhanced SimCSE (ESimCSE) :tada:
- 03/01/2022: Added Mirror-Bert
- 02/01/2022: Added Contrastive Tension (CT)

## Usage:

### Instalation of requirements

```
pip install -r requirements.txt
```

### BERT-flow Training 

Training a bert-base-uncased BERT-flow model using some training text file text_file.txt
```
python train_flow.py --model_name_or_path bert-base-uncased --train_file text_file.txt --output_dir result --num_train_epochs 1 --max_seq_length 512 --per_device_train_batch_size 64
```

### BERT-flow Evaluation

Running evaluation on [SentEval](https://github.com/facebookresearch/SentEval), simply provide the path of the model trained in the previous script to the evaluation shells script:

```
sh eval_flow.sh <path to BERT-flow model>
```


### BERT Contrastive Tension Training 

Training a bert-base-uncased contrastive tensions model using some training text file text_file.txt
```
python train_ct.py --model_name_or_path bert-base-uncased --train_file text_file.txt --output_dir result --num_train_epochs 1 --max_seq_length 512 --per_device_train_batch_size 64
```

### BERT Contrastive Tensions Evaluation

Running evaluation on [SentEval](https://github.com/facebookresearch/SentEval), simply provide the path of the model trained in the previous script to the evaluation shells script:

```
sh eval_ct.sh <path to contrastive tensions model>
```

### Enhanced SimCSE Training

Training a bert-base-uncased ESimCDSE model using some training text file text_file.txt
```
python train_esimcse.py --model_name_or_path bert-base-uncased --train_file text_file.txt --output_dir result --num_train_epochs 1 --max_seq_length 50 --per_device_train_batch_size 64
```


# PyTorch BERT-flow Reimplementation

Simple and lean PyTorch BERT-flow implementation based on the code from [repository](https://github.com/UKPLab/pytorch-bertflow). Reimplementation allows to reproduce the results from original [repository](https://github.com/bohanli/BERT-flow).

Changes:
* Added batching
* Added monitoring (Weights and Biases)
* Added evaluation ([SentEval](https://github.com/facebookresearch/SentEval)) script

## Usage:


### Training 

Training a bert-base-uncased BERT-flow model using some training text file text_file.txt
```
python train.py --model_name_or_path bert-base-uncased --train_file text_file.txt --output_dir result --num_train_epochs 1 --max_seq_length 512 --per_device_train_batch_size 64
```

### Evaluation

Running evaluation on [SentEval](https://github.com/facebookresearch/SentEval), simply provide the path of the model trained in the previous script to the evaluation shells script:

```
sh eval.sh <path to BERT-flow model>
```

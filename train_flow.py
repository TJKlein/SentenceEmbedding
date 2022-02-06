#import senteval
import random
import sys
PATH_TO_SENTEVAL = './SentEval'
PATH_TO_DATA = './SentEval/data'
# Import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval
from tflow_utils import TransformerGlow, AdamWeightDecayOptimizer
from transformers import AutoTokenizer
import argparse
import os
import wandb
from datasets import load_dataset
import sys
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
import transformers
from tqdm import tqdm, trange
import numpy as np
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_MASKED_LM_MAPPING,
    AutoConfig,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    DataCollatorWithPadding,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
    EvalPrediction,
    BertModel,
    BertForPreTraining,
    RobertaModel
)

PATH_TO_SENTEVAL = './SentEval'
PATH_TO_DATA = './SentEval/data'

parser = argparse.ArgumentParser()
parser.add_argument("--no_cuda", action='store_true',
                    help="Avoid using CUDA when available")
parser.add_argument('--model_name_or_path', required=True, type=str, help='Mode for filename to Huggingface model')
parser.add_argument('--train_file', type=str, help='Input training file (text)')
parser.add_argument('--output_dir', required=True, type=str, help='Path where result should be stored')
parser.add_argument('--preprocessing_num_workers', type=int, default=1, help='Number of worker threads for processing data')
parser.add_argument('--overwrite_cache', type=bool, default=False, help='Indicate whether cached features should be overwritten')
parser.add_argument('--pad_to_max_length', type=bool, default=True, help='Indicate whether tokens sequence should be padded')
parser.add_argument('--max_seq_length', type=int, default=32, help='Input max sequence length in tokens')
parser.add_argument('--overwrite_output_dir', type=bool, default=True, help="If data in output directory should be overwritten if already existing.")
parser.add_argument('--learning_rate', type=float,
                    default=1e-3, help='SGD learning rate')
parser.add_argument('--num_train_epochs', type=int, default=3,
                    help='Number of trainin epochs')
parser.add_argument('--per_device_train_batch_size', type=int, default=64, help='Batch size for training')
parser.add_argument('--description', type=str, help='Experiment description')
parser.add_argument('--tags', type=str, help='Annotation tags for wandb, comma separated')
parser.add_argument('--eval_steps', type=int, default=250, help='Frequency of model selection evaluation')
parser.add_argument('--metric_for_best_model', type=str, choices=['sickr_spearman','stsb_spearman'], default='stsb_spearman', help='Metric for model selection')
parser.add_argument('--seed', type=int, default=48, help='Random seed for reproducability')
parser.add_argument("--pooler", type=str,
                        choices=['mean', 'max', 'cls', 'first-last-avg'],
                    default='first-last-avg',
                        help="Which pooler to use")
parser.add_argument("--local_rank",
                    type=int,
                    default=-1,
                    help="local_rank for distributed training on gpus")


def set_seed(seed: int):
    """
    Helper function for reproducible behavior to set the seed in ``random``, ``numpy``, ``torch`` and/or ``tf`` (if
    installed).

    Args:
        seed (:obj:`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

if __name__ == '__main__':
    
    FLAGS = parser.parse_args()
    
    
    
    if not (FLAGS.tags is None):
        FLAGS.tags = [item for item in FLAGS.tags.split(',')]

    if not(FLAGS.tags is None):
        wandb.init(project=FLAGS.description, tags=FLAGS.tags)

    else:
        wandb.init(project=FLAGS.description)

    wandb.config.update({"Command Line": 'python '+' '.join(sys.argv[0:])})

    if not(wandb.run.name is None):
        output_name = wandb.run.name
    else:
        output_name = 'dummy-run'
    
    FLAGS.output_dir = os.path.join(
        FLAGS.output_dir, output_name)
    
    
    if (
        os.path.exists(FLAGS.output_dir)
        and os.listdir(FLAGS.output_dir)
        and not FLAGS.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({FLAGS.output_dir}) already exists and is not empty."
            "Use --overwrite_output_dir to overcome."
        )


    if FLAGS.local_rank == -1 or FLAGS.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        FLAGS.n_gpu = torch.cuda.device_count()
    
    # make sure we can make all the experiment results reproducible
    set_seed(FLAGS.seed)
    bertflow = TransformerGlow(FLAGS.model_name_or_path, pooling=FLAGS.pooler)  # pooling could be 'mean', 'max', 'cls' or 'first-last-avg' (mean pooling over the first and the last layers)
    tokenizer = AutoTokenizer.from_pretrained(FLAGS.model_name_or_path)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters= [
        {
            "params": [p for n, p in bertflow.glow.named_parameters()  \
                            if not any(nd in n for nd in no_decay)],  # Note only the parameters within bertflow.glow will be updated and the Transformer will be freezed during training.
            "weight_decay": 0.01,
        },
        {
            "params": [p for n, p in bertflow.glow.named_parameters()  \
                            if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamWeightDecayOptimizer(
        params=optimizer_grouped_parameters, 
        lr=FLAGS.learning_rate, 
        eps=1e-6,
    )

    datasets = load_dataset(path = os.path.split(os.path.abspath(FLAGS.train_file))[0], data_files=os.path.split(os.path.abspath(FLAGS.train_file))[1], cache_dir="./data/")
    column_names = datasets["train"].column_names

    
     # Unsupervised datasets
    sent0_cname = column_names[0]
    
    def prepare_features(examples):
        # padding = longest (default)
        #   If no sentence in the batch exceed the max length, then use
        #   the max sentence length in the batch, otherwise use the 
        #   max sentence length in the argument and truncate those that
        #   exceed the max length.
        # padding = max_length (when pad_to_max_length, for pressure test)
        #   All sentences are padded/truncated to data_args.max_seq_length.
        total = len(examples[sent0_cname])

        # Avoid "None" fields 
        for idx in range(total):
            if examples[sent0_cname][idx] is None:
                examples[sent0_cname][idx] = " "
          
        sentences = examples[sent0_cname]

        
        sent_features = tokenizer(
            sentences,
            max_length=FLAGS.max_seq_length,
            truncation=True,
            padding="max_length" if FLAGS.pad_to_max_length else False,
        )

      
            
        return sent_features
    
    
    train_dataset = datasets["train"].map(
            prepare_features,
            batched=True,
            num_proc=FLAGS.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not FLAGS.overwrite_cache,
        )    

    all_input_ids = torch.tensor(
        [f for f in train_dataset['input_ids']], dtype=torch.long)
    all_attention_mask = torch.tensor(
        [f for f in train_dataset['attention_mask']], dtype=torch.long)
    
    train_data = TensorDataset(all_input_ids, all_attention_mask)

    train_sampler = RandomSampler(train_data)

    train_dataloader = DataLoader(
        train_data, sampler=train_sampler, batch_size=FLAGS.per_device_train_batch_size)
    
    bertflow.to(device)
    bertflow.train()

    # variables for monitoring
    global_steps = 0
    best_metric = None
    # trai nthe model 
    for it in trange(int(FLAGS.num_train_epochs), desc="Epoch"):

        for step, batch in enumerate((train_dataloader)):
            global_steps = global_steps + 1
            input_ids, attention_mask = (tens.to(device) for tens in batch)

            z, loss = bertflow(input_ids,attention_mask, return_loss=True)  # Here z is the sentence embedding
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            wandb.log({'loss': loss.item()})

            if global_steps % FLAGS.eval_steps == 1 and global_steps > 1:

                params = {'task_path': PATH_TO_DATA,
                          'usepytorch': True, 'kfold': 5}
                params['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
                                'tenacity': 3, 'epoch_size': 2}

                # SentEval prepare and batcher
                def prepare(params, samples):
                    return
                def batcher(params, batch, max_length=None):
                    # Handle rare token encoding issues in the dataset
                    if len(batch) >= 1 and len(batch[0]) >= 1 and isinstance(batch[0][0], bytes):
                        batch = [[word.decode('utf-8') for word in s] for s in batch]

                    sentences = [' '.join(s) for s in batch]

                    # Tokenization
                    if max_length is not None:
                        batch = tokenizer.batch_encode_plus(
                            sentences,
                            return_tensors='pt',
                            padding=True,
                            max_length=max_length,
                            truncation=True
                        )
                    else:
                        batch = tokenizer.batch_encode_plus(
                            sentences,
                            return_tensors='pt',
                            padding=True,
                        )

                    # Move to the correct device
                    for k in batch:
                        batch[k] = batch[k].to(device)

                    # Get raw embeddings
                    with torch.no_grad():
                        del batch['token_type_ids']
                        z = bertflow(**batch)
                        return z.cpu()

                results = {}

                for task in ['STSBenchmark', 'SICKRelatedness']:
                    se = senteval.engine.SE(params, batcher, prepare)
                    result = se.eval(task)
                    results[task] = result

                stsb_spearman = results['STSBenchmark']['dev']['spearman'][0]
                sickr_spearman = results['SICKRelatedness']['dev']['spearman'][0]
                wandb.log({"eval_stsb_spearman": stsb_spearman,
                          "eval_sickr_spearman": sickr_spearman, "eval_avg_sts": (stsb_spearman + sickr_spearman) / 2})
                metrics = {"eval_stsb_spearman": stsb_spearman,
                          "eval_sickr_spearman": sickr_spearman, "eval_avg_sts": (stsb_spearman + sickr_spearman) / 2}

                # Determine the new best metric / best model checkpoint
                if metrics is not None and FLAGS.metric_for_best_model is not None:
                    metric_to_check = FLAGS.metric_for_best_model
                    if not metric_to_check.startswith("eval_"):
                        metric_to_check = f"eval_{metric_to_check}"
                    metric_value = metrics[metric_to_check]

                    operator = np.greater
                    if (
                        best_metric is None
                        or operator(metric_value, best_metric)
                    ):
                        # update the best metric
                        best_metric = metric_value

                        # save the best (intermediate) model
                        bertflow.save_pretrained(
                            FLAGS.output_dir)  # Save model
                        #bertflow = TransformerGlow.from_pretrained(FLAGS.output_dir)  # Load model
                        tokenizer.save_pretrained(FLAGS.output_dir)

        

   

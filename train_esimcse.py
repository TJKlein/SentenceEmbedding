#!/usr/bin/env python
# from src.drophead import set_drophead
# from src.contrastive_learning import ContrastiveLearningPairwise
# from src.data_loader import ContrastiveLearningDataset
# from src.mirror_bert import MirrorBERT
from torch.cuda.amp import autocast, GradScaler
import sys
PATH_TO_SENTEVAL = './SentEval'
PATH_TO_DATA = './SentEval/data'
# Import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval
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
import numpy as np
from tqdm import tqdm, trange
import transformers
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
import torch
from datasets import load_dataset
import wandb
import os
import argparse
from transformers import AutoTokenizer
from src.ESimCSE import ESimCSEModel, MomentumEncoder, MultiNegativeRankingLoss, Similarity
from src.data_loader_esimcse import TrainDataset, CollateFunc
#import senteval
import random

parser = argparse.ArgumentParser()
parser.add_argument("--no_cuda", action='store_true',
                    help="Avoid using CUDA when available")
parser.add_argument('--model_name_or_path', required=True,
                    type=str, help='Mode for filename to Huggingface model')
parser.add_argument('--train_file', type=str,
                    help='Input training file (text)')
parser.add_argument('--output_dir', required=True, type=str,
                    help='Path where result should be stored')
parser.add_argument('--preprocessing_num_workers', type=int,
                    default=1, help='Number of worker threads for processing data')
parser.add_argument('--overwrite_cache', type=bool, default=False,
                    help='Indicate whether cached features should be overwritten')
parser.add_argument('--pad_to_max_length', type=bool, default=True,
                    help='Indicate whether tokens sequence should be padded')
parser.add_argument('--max_seq_length', type=int, default=32,
                    help='Input max sequence length in tokens')
parser.add_argument('--overwrite_output_dir', type=bool, default=True,
                    help="If data in output directory should be overwritten if already existing.")
parser.add_argument('--dropout', type=float, default=0.1, help="dropout rate for encoder")
parser.add_argument('--dup_level', type=str, default="subword", help="Level of duplication: word, subword")
parser.add_argument('--learning_rate', default=3e-5,
                    type=float, help='SGD learning rate')
parser.add_argument('--weight_decay', default=0.01, type=float)
parser.add_argument('--gamma', default=0.95, type=float, help="Moment encoder factor")
parser.add_argument('--dup_rate', type=float, default=0.32)
parser.add_argument('--temp', type=float, default=0.05, help="Temperature for similarity")
parser.add_argument("--q_size", type=int, default=80)
parser.add_argument('--num_train_epochs', type=int, default=1,
                    help='Number of trainin epochs')
parser.add_argument('--per_device_train_batch_size', type=int,
                    default=32, help='Batch size for training')
parser.add_argument('--description', type=str, help='Experiment description')
parser.add_argument('--tags', type=str,
                    help='Annotation tags for wandb, comma separated')
parser.add_argument('--eval_steps', type=int, default=250,
                    help='Frequency of model selection evaluation')
parser.add_argument('--metric_for_best_model', type=str, choices=[
                    'sickr_spearman', 'stsb_spearman'], default='stsb_spearman', help='Metric for model selection')
parser.add_argument('--seed', type=int, default=48,
                    help='Random seed for reproducability')
parser.add_argument("--pooler", type=str,
                    choices=['mean', 'max', 'cls', 'first-last-avg'],
                    default='first-last-avg',
                    help="Which pooler to use")
parser.add_argument("--local_rank",
                    type=int,
                    default=-1,
                    help="local_rank for distributed training on gpus")
parser.add_argument('--amp', action="store_true",
                    help="automatic mixed precision training")


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


def get_bert_input(source, device):
    input_ids = source.get('input_ids').to(device)
    attention_mask = source.get('attention_mask').to(device)
    token_type_ids = source.get('token_type_ids').to(device)
    return input_ids, attention_mask, token_type_ids


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


    # mixed precision training 
    if FLAGS.amp:
        scaler = GradScaler()
    else:
        scaler = None

    # make sure we can make all the experiment results reproducible
    set_seed(FLAGS.seed)
    # pooling could be 'mean', 'max', 'cls' or 'first-last-avg' (mean pooling over the first and the last layers)
    tokenizer = AutoTokenizer.from_pretrained(FLAGS.model_name_or_path)

    datasets = load_dataset(path=os.path.split(os.path.abspath(FLAGS.train_file))[
                            0], data_files=os.path.split(os.path.abspath(FLAGS.train_file))[1], cache_dir="./data/")
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

        return {"sentence": sentences}

    train_data = datasets["train"].map(
        prepare_features,
        batched=True,
        num_proc=FLAGS.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=not FLAGS.overwrite_cache,
    )

    train_dataset = TrainDataset(train_data['sentence'])

    # all_input_ids = torch.tensor(
    #     [f for f in train_dataset['input_ids']], dtype=torch.long)
    # all_attention_mask = torch.tensor(
    #     [f for f in train_dataset['attention_mask']], dtype=torch.long)

    # train_data = TensorDataset(all_input_ids, all_attention_mask)

    # train_sampler = RandomSampler(train_data)

    train_call_func = CollateFunc(
        tokenizer, repetition=FLAGS.dup_level, max_len=FLAGS.max_seq_length, q_size=FLAGS.q_size, dup_rate=FLAGS.dup_rate)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=FLAGS.per_device_train_batch_size, num_workers=12,
                                  collate_fn=train_call_func)

    model = ESimCSEModel(pretrained_model=FLAGS.model_name_or_path, pooler_type=FLAGS.pooler, dropout=FLAGS.dropout).to(device)
    momentum_encoder = MomentumEncoder(
        FLAGS.model_name_or_path, FLAGS.pooler).to(device)

    optimizer = torch.optim.AdamW([{'params': model.parameters()}, ],
                                  lr=FLAGS.learning_rate, weight_decay=FLAGS.weight_decay)

    model.train()
    #ESimCSELoss = MultiNegativeRankingLoss()
    #criterion = ESimCSELoss.multi_negative_ranking_loss
    criterion = torch.nn.CrossEntropyLoss()
    # variables for monitoring
    global_steps = 0
    best_metric = None

    sim = Similarity(temp=FLAGS.temp)
    # trai nthe model
    for it in trange(int(FLAGS.num_train_epochs), desc="Epoch"):

        for step, (batch_src_source, batch_pos_source, batch_neg_source)  in enumerate(tqdm(train_dataloader, desc="Steps")):
            model.train()

            global_steps = global_steps + 1
            input_ids_src, attention_mask_src, token_type_ids_src = get_bert_input(
                batch_src_source, device)
            input_ids_pos, attention_mask_pos, token_type_ids_pos = get_bert_input(
                batch_pos_source, device)

            neg_out = None
            if batch_neg_source:
                input_ids_neg, attention_mask_neg, token_type_ids_neg = get_bert_input(
                    batch_neg_source, device)
                
                if FLAGS.amp:
                    with autocast():
                        neg_out = momentum_encoder(
                        input_ids_neg, attention_mask_neg, token_type_ids_neg)
                else:
                    neg_out = momentum_encoder(
                        input_ids_neg, attention_mask_neg, token_type_ids_neg)

            if FLAGS.amp:
                with autocast():
                    src_out = model(input_ids_src, attention_mask_src, token_type_ids_src)
                    pos_out = model(input_ids_pos, attention_mask_pos, token_type_ids_pos)
            else:
                src_out = model(input_ids_src, attention_mask_src, token_type_ids_src)
                pos_out = model(input_ids_pos, attention_mask_pos, token_type_ids_pos)


            # print(embed_src.shape, embed_pos.shape)
            cos_sim = sim(src_out.unsqueeze(1), pos_out.unsqueeze(0))

            # Hard negative
            if neg_out is not None:
                tmp = sim(src_out.unsqueeze(1), neg_out.unsqueeze(0))
                cos_sim = torch.cat([cos_sim, tmp], 1)

            #loss = criterion(src_out, pos_out, neg_out)
            labels = torch.arange(cos_sim.size(0)).long().to(device)
            loss = criterion(cos_sim, labels)
            optimizer.zero_grad()

            if FLAGS.amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            #  Momentum Contrast Encoder Update
            for encoder_param, moco_encoder_param in zip(model.parameters(), momentum_encoder.parameters()):
                # print("--", moco_encoder_param.data.shape, encoder_param.data.shape)
                moco_encoder_param.data = FLAGS.gamma \
                    * moco_encoder_param.data \
                    + (1. - FLAGS.gamma) * encoder_param.data

            wandb.log({'train/loss': loss.item()})

            if global_steps % FLAGS.eval_steps == 1 and global_steps > 1:

                model.eval()

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
                        batch = [[word.decode('utf-8')
                                  for word in s] for s in batch]

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
                        #el batch['token_type_ids']
                        z = model(**batch)
                        return z.cpu()

                results = {}

                for task in ['STSBenchmark', 'SICKRelatedness']:
                    se = senteval.engine.SE(params, batcher, prepare)
                    result = se.eval(task)
                    results[task] = result

                stsb_spearman = results['STSBenchmark']['dev']['spearman'][0]
                sickr_spearman = results['SICKRelatedness']['dev']['spearman'][0]
                wandb.log({"eval/stsb_spearman": stsb_spearman,
                          "eval/sickr_spearman": sickr_spearman, "eval/avg_sts": (stsb_spearman + sickr_spearman) / 2})
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
                        model.save_model(
                            FLAGS.output_dir)  # Save model
                        tokenizer.save_pretrained(FLAGS.output_dir)





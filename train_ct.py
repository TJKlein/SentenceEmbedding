#import senteval
import random
import sys
import argparse
import os
import wandb
import numpy as np
from datasets import load_dataset
import sys
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers import LoggingHandler, models, util, InputExample
from sentence_transformers import losses
import os
import gzip
import csv
from datetime import datetime
from sentence_transformers.evaluation import SentenceEvaluator

from SentenceTransformer import SentenceTransformer

# Set PATHs
PATH_TO_SENTEVAL = './SentEval'
PATH_TO_DATA = './SentEval/data'

# Import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval

parser = argparse.ArgumentParser()
parser.add_argument('--metric_for_best_model', type=str, choices=['sickr_spearman','stsb_spearman'], default='stsb_spearman', help='Metric for model selection')
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
                    default=1e-5, help='SGD learning rate')
parser.add_argument("--pooler", type=str,
                        choices=['mean', 'max', 'cls'],
                    default='mean',
                        help="Which pooler to use")
parser.add_argument('--num_train_epochs', type=int, default=3,
                    help='Number of trainin epochs')
parser.add_argument('--per_device_train_batch_size', type=int, default=64, help='Batch size for training')
parser.add_argument('--description', type=str, help='Experiment description')
parser.add_argument('--tags', type=str, help='Annotation tags for wandb, comma separated')
parser.add_argument('--eval_steps', type=int, default=250, help='Frequency of model selection evaluation')
parser.add_argument('--seed', type=int, default=48, help='Random seed for reproducability')
parser.add_argument("--local_rank",
                    type=int,
                    default=-1,
                    help="local_rank for distributed training on gpus")






class SentEvalSimilarityEvaluator(SentenceEvaluator):
    """
    Evaluate a model based on the similarity of the embeddings by calculating the Spearman and Pearson rank correlation
    in comparison to the gold standard labels.
    The metrics are the cosine similarity as well as euclidean and Manhattan distance
    The returned score is the Spearman correlation with a specified metric.
    The results are written in a CSV. If a CSV already exists, then values are appended.
    """
    def __init__(self, FLAGS):
        """
        Constructs an evaluator based for the dataset
        The labels need to indicate the similarity between the sentences.
        :param sentences1:  List with the first sentence in a pair
        :param sentences2: List with the second sentence in a pair
        :param scores: Similarity score between sentences1[i] and sentences2[i]
        :param write_csv: Write results to a CSV file
        """
        self.params = {'task_path': PATH_TO_DATA,
                          'usepytorch': True, 'kfold': 5}
        self.params['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
                                'tenacity': 3, 'epoch_size': 2}
        self.FLAGS = FLAGS

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        

        # SentEval prepare and batcher
        def prepare(params, samples):
            return
        def batcher(params, batch, max_length=None):
            # Handle rare token encoding issues in the dataset
            if len(batch) >= 1 and len(batch[0]) >= 1 and isinstance(batch[0][0], bytes):
                batch = [[word.decode('utf-8') for word in s] for s in batch]

            sentences = [' '.join(s) for s in batch]


            # Get raw embeddings
            with torch.no_grad():
                z = model.encode(sentences,  batch_size = 128, show_progress_bar=False, convert_to_numpy=True)
                return z
            
        results = {}

        for task in ['STSBenchmark', 'SICKRelatedness']:
            se = senteval.engine.SE(self.params, batcher, prepare)
            result = se.eval(task)
            results[task] = result

        stsb_spearman = results['STSBenchmark']['dev']['spearman'][0]
        sickr_spearman = results['SICKRelatedness']['dev']['spearman'][0]
        
        metrics = {"eval_stsb_spearman": stsb_spearman,
                  "eval_sickr_spearman": sickr_spearman, "eval_avg_sts": (stsb_spearman + sickr_spearman) / 2}
        
        log_metrics = {"eval/stsb_spearman": stsb_spearman,
                  "eval/sickr_spearman": sickr_spearman, "eval/avg_sts": (stsb_spearman + sickr_spearman) / 2}

        wandb.log(log_metrics)
        # Determine the new best metric / best model checkpoint
        if metrics is not None and self.FLAGS.metric_for_best_model is not None:
            metric_to_check = self.FLAGS.metric_for_best_model
            if not metric_to_check.startswith("eval_"):
                metric_to_check = f"eval_{metric_to_check}"
            metric_value = metrics[metric_to_check]
            return metric_value
        else:
            raise ValueError("Unknown main_similarity value")

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
    ################# Download and load STSb #################
    data_folder = 'data/stsbenchmark'
    sts_dataset_path = f'{data_folder}/stsbenchmark.tsv.gz'

    if not os.path.exists(sts_dataset_path):
        util.http_get('https://sbert.net/datasets/stsbenchmark.tsv.gz', sts_dataset_path)

    datasets = load_dataset(path = os.path.split(os.path.abspath(FLAGS.train_file))[0], data_files={'train': os.path.split(os.path.abspath(FLAGS.train_file))[1]}, cache_dir="./data/")
    

#     dev_samples = []
#     test_samples = []
#     with gzip.open(sts_dataset_path, 'rt', encoding='utf8') as fIn:
#         reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
#         for row in reader:
#             score = float(row['score']) / 5.0  # Normalize score to range 0 ... 1
#             inp_example = InputExample(texts=[row['sentence1'], row['sentence2']], label=score)

#             if row['split'] == 'dev':
#                 dev_samples.append(inp_example)
#             elif row['split'] == 'test':
#                 test_samples.append(inp_example)

#     dev_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, name='sts-dev')
    #test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_samples, name='sts-test')
    
    dev_evaluator = SentEvalSimilarityEvaluator(FLAGS = FLAGS)

    ################# Intialize an SBERT model #################
    word_embedding_model = models.Transformer(FLAGS.model_name_or_path, max_seq_length=FLAGS.max_seq_length)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode=FLAGS.pooler)
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])


    pos_neg_ratio = 8 
    # For ContrastiveTension we need a special data loader to construct batches with the desired properties
    train_dataloader =  losses.ContrastiveTensionDataLoader(datasets["train"]['text'], batch_size=FLAGS.per_device_train_batch_size, pos_neg_ratio=pos_neg_ratio)

    # As loss, we losses.ContrastiveTensionLoss
    train_loss = losses.ContrastiveTensionLoss(model)


    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=dev_evaluator,
        epochs=FLAGS.num_train_epochs,
        evaluation_steps=FLAGS.eval_steps,
        weight_decay=0,
        warmup_steps=0,
        optimizer_class=torch.optim.RMSprop,
        optimizer_params={'lr': FLAGS.learning_rate},
        output_path=FLAGS.output_dir,
        use_amp=False    #Set to True, if your GPU has optimized FP16 cores
    )

        

   

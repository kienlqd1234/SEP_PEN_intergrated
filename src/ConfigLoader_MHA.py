#!/usr/local/bin/python
import logging
import logging.config
import yaml
import itertools
import os
import io
import json
import sys


class PathParser:

    def __init__(self, config_path):
        self.root = './'
        self.log = os.path.join(self.root, config_path['log'])

        self.data = os.path.join(self.root, config_path['data'])
        self.res = os.path.join(self.root, config_path['res'])
        self.graphs = os.path.join(self.root, config_path['graphs'])
        self.checkpoints = os.path.join(self.root, config_path['checkpoints'])

        self.glove = os.path.join(self.res, config_path['glove'])

        self.retrieved = os.path.join(self.data, config_path['tweet_retrieved'])
        self.preprocessed = os.path.join(self.data, config_path['tweet_preprocessed'])
        self.movement = os.path.join(self.data, config_path['price'])
        self.vocab = os.path.join(self.res, config_path['vocab_tweet'])

# Default attention config
DEFAULT_ATTENTION_CONFIG = {
    'num_heads': 8,
    'head_size': 64,
    'dropout': 0.1,
    'use_layer_norm': True,
    'use_residual': True,
    'text_attention': {
        'enabled': True,
        'causal': False
    },
    'cross_attention': {
        'enabled': True,
        'temperature': 1.0
    }
}

def load_attention_config(config_model):
    """Load attention configuration with defaults"""
    if 'attention' not in config_model:
        config_model['attention'] = {}
    
    # Merge with defaults
    attention_config = config_model['attention']
    for key, value in DEFAULT_ATTENTION_CONFIG.items():
        if key not in attention_config:
            attention_config[key] = value
        elif isinstance(value, dict):
            for sub_key, sub_value in value.items():
                if sub_key not in attention_config[key]:
                    attention_config[key][sub_key] = sub_value
    
    return attention_config

config_fp = os.path.join(os.path.dirname(__file__), 'config_MHA.yml')
#config = yaml.load(file(config_fp, 'r'))
with open(config_fp, 'r') as config_file:
    config = yaml.load(config_file, Loader=yaml.FullLoader)
config_model = config['model']

# Load attention config with defaults
attention_config = load_attention_config(config_model)

# Add attention config to model config
config_model['attention'] = attention_config

dates = config['dates']

config_stocks = config['stocks']  # a list of lists
stock_symbols = list(itertools.chain.from_iterable(config_stocks.values()))
#list_of_lists = [config_stocks[key] for key in config_stocks]
#stock_symbols = list(itertools.chain.from_iterable(list_of_lists))
#stock_symbols = config_stocks
ss_size = len(stock_symbols)

path_parser = PathParser(config_path=config['paths'])

# logger
logger = logging.getLogger('my_logger')
logger.setLevel(logging.DEBUG)

log_dir = os.path.dirname(path_parser.log)
os.makedirs(log_dir, exist_ok=True)  # Create the log directory if it doesn't exist

log_fp = os.path.join(path_parser.log, '{0}.log'.format('model'))
file_handler = logging.FileHandler(log_fp)
console_handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

with io.open(str(path_parser.vocab), 'r', encoding='utf-8') as vocab_f:
    vocab = json.load(vocab_f)
    vocab_size = len(vocab) + 1  # for unk

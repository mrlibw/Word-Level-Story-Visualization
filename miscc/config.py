from __future__ import division
from __future__ import print_function

import os.path as osp
import numpy as np
from easydict import EasyDict as edict


__C = edict()
cfg = __C

# Dataset name: flowers, birds
__C.DATASET_NAME = 'pororo'
__C.EMBEDDING_TYPE = 'cnn-rnn'
__C.CONFIG_NAME = ''
__C.GPU_ID = '0'
__C.CUDA = True
__C.WORKERS = 6
__C.VIDEO_LEN = 5
__C.NET_G = ''
__C.NET_D = ''
__C.STAGE1_G = ''
__C.DATA_DIR = ''
__C.VIS_COUNT = 64

__C.USE_SEQ_CONSISTENCY = False
__C.CONSISTENCY_RATIO = 1.0
__C.IMAGE_RATIO = 5.0
__C.RECONSTRUCT_LOSS = 1.0
__C.EVALUATE_FID_SCORE = False
__C.CASCADE_MODEL = False
__C.Z_DIM = 100
__C.IMSIZE = 64
__C.SESIZE = 64
__C.STAGE = 1

__C.LABEL_NUM = 9

# Training options
__C.TRAIN = edict()
__C.TRAIN.FLAG = True
__C.TRAIN.IM_BATCH_SIZE = 64
__C.TRAIN.ST_BATCH_SIZE = 64
__C.TRAIN.MAX_EPOCH = 600
__C.TRAIN.SNAPSHOT_INTERVAL = 50
__C.TRAIN.PRETRAINED_MODEL = ''
__C.TRAIN.PRETRAINED_EPOCH = 600
__C.TRAIN.LR_DECAY_EPOCH = 600
__C.TRAIN.DISCRIMINATOR_LR = 2e-4
__C.TRAIN.GENERATOR_LR = 2e-4
__C.TRAIN.COEFF = edict()
__C.TRAIN.COEFF.KL = 2.0
__C.TRAIN.TEXT_ENCODER = ''
__C.TRAIN.SMOOTH = edict()
__C.TRAIN.GAMMA = 10.0
__C.TRAIN.LAMBDA = 1.0
# Modal options
__C.GAN = edict()
__C.GAN.CONDITION_DIM = 124
__C.GAN.Z_DIM = 100
__C.GAN.DF_DIM = 124
__C.GAN.GF_DIM = 256
__C.GAN.R_NUM = 4

__C.TEXT = edict()
__C.TEXT.DIMENSION = 356
__C.TEXT.WORDS_NUM = 20
__C.RNN_TYPE = 'LSTM' 
__C.TRAIN.ENCODER_LR = 2e-4
__C.TRAIN.RNN_GRAD_CLIP = 0.25

def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return

    for k, v in a.items():
        # a must specify keys that are in b
        #if not b.has_key(k):
        if k not in b.keys():
            raise KeyError('{} is not a valid config key'.format(k))

        # the types must match, too
        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            else:
                raise ValueError(('Type mismatch ({} vs. {}) '
                                  'for config key: {}').format(type(b[k]),
                                                               type(v), k))

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print('Error under config key: {}'.format(k))
                raise
        else:
            b[k] = v


def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f))

    _merge_a_into_b(yaml_cfg, __C)

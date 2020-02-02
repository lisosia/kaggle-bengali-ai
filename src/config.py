import os
import yaml
from pathlib import Path
import torch

from easydict import EasyDict


def _get_default_config():
    c = EasyDict()

    c.submission=False

    c.batch_size=128
    c.device='cuda:0' if torch.cuda.is_available() else 'cpu'
    c.out='.'
    c.image_size=128
    c.arch='pretrained'
    c.model_name='se_resnext50_32x4d'

    c.datadir = Path('../input/bengaliai-cv19')
    c.featherdir = Path('../input/bengaliaicv19feather')
    c.pngdir = Path('../input/bengaliai-cv19-png')
    c.outdir = Path('.')

    c.n_epoch = 10
    c.num_workers = 4

    return c


# borrowed from:
# https://github.com/bamps53/kaggle-autonomous-driving2019/blob/master/config/base.py
def _merge_config(src, dst):
    if not isinstance(src, EasyDict):
        raise

    for k, v in src.items():
        if isinstance(v, EasyDict):
            _merge_config(src[k], dst[k])
        else:
            dst[k] = v


global_config = None  # set_yaml in main()

def get_config(config_path=None):
    global global_config
    if global_config is not None:
        return global_config
    elif config_path is None:
        raise Exception("gibe config_path arg when first call ")

    config = _get_default_config()

    with open(config_path, 'r') as fid:
        yaml_config = EasyDict(yaml.load(fid, Loader=yaml.SafeLoader))

    _merge_config(yaml_config, config)

    config.outdir = f'../run/{os.path.basename(config_path)}'
    os.makedirs(config.outdir)

    global_config = config

    print("config: ", config)
    return config


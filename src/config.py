from pathlib import Path
import torch

class _Conf():
    debug=True
    submission=False
    batch_size=32
    device='cuda:0' if torch.cuda.is_available() else 'cpu'
    out='.'
    image_size=64
    arch='pretrained'
    model_name='se_resnext50_32x4d'

    datadir = Path('../input/bengaliai-cv19')
    featherdir = Path('../input/bengaliaicv19feather')
    outdir = Path('.')

    n_epoch = 10

    def __init__(self):
        raise


def _get_default_config():
    return _Conf


def load_config(yaml_path=None):
    return _get_default_config()

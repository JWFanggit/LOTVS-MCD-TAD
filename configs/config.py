import argparse
import yaml
from easydict import EasyDict as edict


def pase_args():
    # parser = argparse.ArgumentParser(description='abnormal detection')
    #
    # parser.add_argument('--author', type=str, default='ydx', help='.....')
    #
    # args = parser.parse_args()

    with open('configs/my_train.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        config = edict(config)
    return config


args = pase_args()



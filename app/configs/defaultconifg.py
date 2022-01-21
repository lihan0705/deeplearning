"""
config for building network
and training hyper parameters
"""
import sys
from typing import List, Optional

try: 
    from typing import TypedDict
except:
    from typing_extensions import TypedDict
from types import SimpleNamespace

class DefaultConfig(object):

    env = 'default'
    model = 'AlexNet'

    train_data_root = './data/train/'
    test_data_root = './data/test1'
    load_model_path = 'checkpoints/model.pth'

    batch_size = 128
    use_gpu = True
    num_workers = 4
    print_freq = 20

    max_epoch = 10
    lr = 0.1
    lr_decay = 0.95
    weight_decay = 1e-4

class SegmentationConfig(object):
    env = 'default'
    model = 'Unet'

    train_data_root = './data/train/'
    test_data_root = './data/test1'
    load_model_path = 'checkpoints/model.pth'

    batch_size = 128
    use_gpu = True
    num_workers = 4
    print_freq = 20

    max_epoch = 10
    lr = 0.1
    lr_decay = 0.95
    weight_decay = 1e-4

    # build net config
    # ToDo: not a good way to build it
    net_double_conv_in_channels = 3
    net_num_classes = 3

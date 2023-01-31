import os
import importlib
import yaml

from configs import base


def load_config(config_name):
    # config = importlib.import_module('configs.' + config_name)
    if os.path.isfile(config_name):
        with open(config_name) as f:
            config = yaml.safe_load(f)
    else:
        with open(os.path.join('configs', config_name + '.yaml'), 'r') as f:
            config = yaml.safe_load(f)
    config = base.Config.copy_from_dict(config)
    return config


def cycle(iterator):
    while True:
        for x in iterator:
            yield x
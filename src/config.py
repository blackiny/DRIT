from easydict import EasyDict as edict

__C = edict()
# Consumers can get config by:
#   from fast_rcnn_config import cfg
cfg = __C

#
# Common
#
# Path where taxonomy.json is stored
__C.DATASET = '../datasets/shapenet_1000.json'


#
# Directories
#
__C.DIR = edict()
#__C.DIR.RENDERING_PATH = '/data2/sunke/ShapeNetRendering/%s/%s/rendering'


#
# Training
#
__C.TRAIN = edict()
__C.TRAIN.N_VIEWS = 5
__C.TRAIN.NUM_RENDERING = 24
__C.TRAIN.RANDOM_NUM_VIEWS = True  # feed in random # views if n_views > 1
__C.TRAIN.AZIMUTH_RANGE = 50
__C.TRAIN.ALTITUDE_RANGE = 90
__C.TRAIN.INCLASS_PAIR_RATIO = 0.3
__C.TRAIN.MAX_MODEL_PER_CLASS = 2000
__C.TRAIN.CLASSES = 'chair,table'
__C.TRAIN.PAIR_EG = False # if one autoencoder for each domain


def _merge_a_into_b(a, b):
  """Merge config dictionary a into config dictionary b, clobbering the
  options in b whenever they are also specified in a.
  """
  if type(a) is not edict:
    return

  for k, v in a.items():
    # a must specify keys that are in b
    if k not in b.keys():
      raise KeyError('{} is not a valid config key'.format(k))

      # the types must match, too
    if type(b[k]) is not type(v):
      raise ValueError(('Type mismatch ({} vs. {}) ''for config key: {}').format(type(b[k]), type(v), k))

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
from easydict import EasyDict as edict

__C = edict()
# Consumers can get config by:
#   from fast_rcnn_config import cfg
cfg = __C

#
# Common
#
__C.DATASET = '../dataset/shapenet_1000.json'


#
# Directories
#
__C.DIR = edict()
# Path where taxonomy.json is stored
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

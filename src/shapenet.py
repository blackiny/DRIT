from config import cfg
import os
import torch.utils.data as data
from PIL import Image
import random
import numpy as np
import json
from collections import OrderedDict

# class dataset_single(data.Dataset):
#   def __init__(self, root_dir, input_dim=3, classes=[], transform=None):
#     self.root_dir = root_dir
#     self.classes = classes
#     self.transform = transform
#     self.input_dim = input_dim
#     imgs = []
#     n_class = 0
#     for _dir in os.listdir(root_dir):
#       _dir_path = os.path.join(root_dir, _dir)
#       if not os.path.isdir(_dir_path):
#         continue
#         for img in os.listdir(_dir_path):
#           img_path = os.path.join(_dir_path, img)
#           if os.path.isfile(img_path):
#             imgs.append(img_path)
#     self.imgs = imgs
#     self.dataset_size = len(imgs)
#     print('load %d images for %d classes' %(len(imgs), n_class))

#   def __getitem__(self, index):
#     data = self.load_img(self.imgs[index], self.input_dim)
#     return data

#   def load_img(self, img_name, input_dim):
#     img = Image.open(img_name).convert('RGB')
#     img = self.transform(img)
#     if input_dim == 1:
#       img = img[0, ...] * 0.299 + img[1, ...] * 0.587 + img[2, ...] * 0.114
#       img = img.unsqueeze(0)
#     return img

#   def __len__(self):
#     return self.dataset_size

class shapenet_unpair(data.Dataset):
  def __init__(self, root_dir, input_dim=3, classes=[], transform=None):
    self.root_dir = root_dir
    self.classes = classes
    self.transform = transform
    self.input_dim = input_dim
    self.inverse_d = {}
    self.class_range = {}
    #load cat info
    cats = json.load(open(cfg.DATASET))
    cats = OrderedDict(sorted(cats.items(), key=lambda x: x[0]))
    imgs = []
    _ind = 0
    n_class = 0
    n_model = 0
    n_views = cfg.TRAIN.N_VIEWS
    for _dir in os.listdir(root_dir):
      if _dir not in cats:
        continue
      if classes and cats[_dir]['cat'] not in classes:
        continue
      _dir_path = os.path.join(root_dir, _dir)
      if not os.path.isdir(_dir_path):
        continue
      n_class += 1
      models = self.get_model_names(_dir_path)
      class_start_ind = _ind
      for model in models:
        ok, img_names = self.get_img_names(_dir, model)
        if not ok:
          continue
        n_model += 1
        if cfg.TRAIN.RANDOM_NUM_VIEWS:
          curr_n_views = np.random.randint(min(n_views, len(img_names))) + 1
        else:
          curr_n_views = min(n_views, len(img_names))
        image_inds = np.random.choice(len(img_names), curr_n_views)
        for ind in image_inds:
          img_path = self.get_img_path(_dir, model, img_names[ind])
          if os.path.isfile(img_path):
            imgs.append(img_path)
            self.inverse_d[_ind] = _dir
            _ind += 1
      class_end_ind = _ind - 1
      self.class_range[_dir] = (class_start_ind, class_end_ind)
      print('load %d images from %d models from class=%s, id=%s' %(
        class_end_ind - class_start_ind + 1, len(models), cats[_dir]['cat'], _dir))
    self.imgs = imgs
    self.dataset_size = len(imgs)
    print('totally load %d images from %d models from %d classes' %(len(self.imgs), n_model, n_class))

  def __getitem__(self, index):
    data_A = self.load_img(self.imgs[index], self.input_dim)
    class_A = self.inverse_d[index]
    rand_ind = -1
    while rand_ind < 0 or rand_ind == index:
      # rand_ind = random.randint(0, self.dataset_size - 1)
      rand_ind = self.get_random_img_index(class_A)
    data_B = self.load_img(self.imgs[rand_ind], self.input_dim)
    return {"A": data_A, "B": data_B}

  def load_img(self, img_name, input_dim):
    # discard alpha channel
    img = Image.open(img_name).convert('RGB')
    img = self.transform(img)
    if input_dim == 1:
      img = img[0, ...] * 0.299 + img[1, ...] * 0.587 + img[2, ...] * 0.114
      img = img.unsqueeze(0)
    return img

  def __len__(self):
    return self.dataset_size
  
  def get_img_path(self, cat_id, model_id, img_name):
    return os.path.join(self.root_dir, cat_id, model_id, 'rendering', img_name)
  
  def get_img_names(self, cat_id, model_id):
    result = []
    meta_path = os.path.join(self.root_dir, cat_id, model_id, 'rendering/rendering_metadata.txt')
    img_names = os.path.join(self.root_dir, cat_id, model_id, 'rendering/renderings.txt')
    with open(img_names, 'r') as f:
      names = [line.strip() for line in f]
    with open(meta_path, 'r') as f:
      metas = [line.strip() for line in f]
    if len(names) != cfg.TRAIN.NUM_RENDERING or len(metas) != cfg.TRAIN.NUM_RENDERING:
      return False, result
    for i in range(cfg.TRAIN.NUM_RENDERING):
      info = metas[i].split()
      if len(info) != 5:
        continue
      az, al = float(info[0]), float(info[1])
      if az > cfg.TRAIN.AZIMUTH_RANGE and az < 360 - cfg.TRAIN.AZIMUTH_RANGE:
        continue
      if al > cfg.TRAIN.ALTITUDE_RANGE:
        continue
      result.append(names[i])
    return True, result

  def get_model_names(self, class_root):
    model_names = [name for name in os.listdir(class_root)
                    if os.path.isdir(os.path.join(class_root, name))]
    return sorted(model_names)

  def get_random_img_index(self, class_id):
    class_start_ind, class_end_ind = self.class_range[class_id]
    rd = random.random()
    if rd <= cfg.TRAIN.INCLASS_PAIR_RATIO:
      ind = random.randint(class_start_ind, class_end_ind)
    else:
      ind = random.randint(0, len(self.imgs) - class_end_ind + class_start_ind - 2)
      if ind >= class_start_ind:
        ind = ind + class_end_ind - class_start_ind + 1
    return ind


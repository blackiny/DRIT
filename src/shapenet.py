from config import cfg
import os
import torch.utils.data as data
from PIL import Image
import random
import numpy as np
import json

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
    #load cat info
    cats = json.load(open(cfg.DATASET))
    cats = OrderedDict(sorted(cats.items(), key=lambda x: x[0]))
    imgs = []
    n_class = 0
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
      for model in models:
        imgs = self.get_img_names()
        if cfg.TRAIN.RANDOM_NUM_VIEWS:
          curr_n_views = np.random.randint(min(n_views, len(imgs))) + 1
        else:
          curr_n_views = min(n_views, len(imgs))
        image_inds = np.random.choice(len(imgs), curr_n_views)
        for ind in image_inds:
          img_path = self.get_img_path(_dir, model, imgs[ind])
          if os.path.isfile(img_path):
            imgs.append(img_path)
            self.inverse_d[img_path] = _dir
    self.imgs = imgs
    self.dataset_size = len(imgs)
    print('load %d images for %d classes' %(len(imgs), n_class))

  def __getitem__(self, index):
    data_A = self.load_img(self.imgs[index], self.input_dim)
    rand_ind = -1
    while rand_ind < 0 or rand_ind == index:
      rand_ind = random.randint(0, self.dataset_size - 1)
    data_B = self.load_img(self.imgs[rand_ind], self.input_dim)
    return {"A": data_A, "B": data_B}

  def load_img(self, img_name, input_dim):
    img = Image.open(img_name).convert('RGB')
    img = self.transform(img)
    if input_dim == 1:
      img = img[0, ...] * 0.299 + img[1, ...] * 0.587 + img[2, ...] * 0.114
      img = img.unsqueeze(0)
    return img

  def __len__(self):
    return self.dataset_size
  
  def get_img_path(self, cat_id, model_id, img_name):
    return os.path.join(self.root_dir, cat_id, model_id, img_name)
  
  def get_img_names(self, cat_id, model_id):
    result = []
    meta_path = os.path.join(self.root_dir, cat_id, model_id, 'rendering_metadata.txt')
    img_names = os.path.join(self.root_dir, cat_id, model_id, 'renderings.txt')
    with open(img_names, 'r') as f:
      names = [line.strip() for line in f]
    with open(meta_path, 'r') as f:
      metas = [line.strip() for line in f]
    if len(names) != cfg.TRAIN.NUM_RENDERING or len(metas) != cfg.TRAIN.NUM_RENDERING:
      return False, result
    for i in range(cfg.TRAIN.NUM_RENDERING):
      info = line.split()
      if len(info) != 5:
        continue
      az, al = info[0], info[1]
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


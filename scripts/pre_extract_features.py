import sys, os
sys.path.insert(0, os.getcwd())
import cv2
import json
import math
import torch
import numpy as np
from tqdm import tqdm
from easydict import EasyDict as edict
from lib.module.backbone import backbones
from lib.data.h5io import H5DataWriter

mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
std = np.array([0.229, 0.224, 0.224]).reshape(3, 1, 1)

def load_image(image_dir, file):
    image_path = os.path.join(image_dir, file)
    image = cv2.imread(image_path)
    if image is None: raise Exception("image %s does not exist" % image_path)
    return image

def preprocess_image(image, height, width):
    h, w, c = image.shape
    if c == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)  # convert to RGB
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = cv2.resize(image, (width, height))  # unify size
    image = image.transpose([2, 0, 1])  # HWC to CHW
    image = image.astype(np.float32)  # convert to float
    image = (image / 255.0 - mean) / std  # normalize

    return image

if __name__ == "__main__":

    config = "configs/lsvrd-resnet101-512.json"
    batch_size = 48

    cfg = edict(json.load(open(config)))
    vcfg = cfg.vision_model

    files = os.listdir(vcfg.image_dir)
    files = [file for file in files if file.endswith(".jpg")]

    fields = [{
        "name": "features",
        "shape": [ vcfg.feature_dim, vcfg.feature_height, vcfg.feature_width ],
        "dtype": "float32"
    }]

    os.makedirs(vcfg.cache_dir, exist_ok=True)
    writer = H5DataWriter(vcfg.cache_dir, cfg.dataset, len(files), 16, fields)
    backbone_cls = backbones[vcfg.backbone]
    cnn = backbone_cls()
    cnn.freeze()
    cnn.cuda()

    print("pre-extracting features with %s ..." % vcfg.backbone)
    pbar = tqdm(total=len(files))
    n_batches = int(math.ceil(len(files)/batch_size))
    for b in range(n_batches):

        batch_files = files[b*batch_size: (b+1)*batch_size]
        images = [ load_image(vcfg.image_dir, file) for file in batch_files ]
        images = [ preprocess_image(image, vcfg.image_height, vcfg.image_width) for image in images ]
        images = [ np.expand_dims(image, axis=0) for image in images ]
        images = np.concatenate(images, axis=0)
        images = torch.from_numpy(images).float().cuda()
        features = cnn(images).data.cpu().numpy()

        for i, file in enumerate(batch_files):
            image_id, _ = os.path.splitext(file)
            feature = features[i]
            writer.put(image_id, feature)

        pbar.update(len(batch_files))

    pbar.close()
    writer.close()
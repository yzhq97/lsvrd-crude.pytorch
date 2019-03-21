import sys, os
sys.path.insert(0, os.getcwd())
import cv2
import json
import math
import torch
import threading
import numpy as np
from tqdm import tqdm
from easydict import EasyDict as edict
from lib.module.backbone import backbones
from lib.data.h5io import H5DataWriter

mean = np.array([0.485, 0.456, 0.406]).reshape([3, 1, 1])
std = np.array([0.229, 0.224, 0.224]).reshape([3, 1, 1])

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

class LoaderThread(threading.Thread):
    def __init__(self, files, height, width, output):
        threading.Thread.__init__(self)
        self.files = files
        self.height = height
        self.width = width
        self.output = output
    def run(self):
        images = [load_image(vcfg.image_dir, file) for file in self.files]
        images = [preprocess_image(image, vcfg.image_height, vcfg.image_width) for image in images]
        images = [np.expand_dims(image, axis=0) for image in images]
        self.output.extend(images)

class WriterThread(threading.Thread):
    def __init__(self, h5_writer, features, files):
        threading.Thread.__init__(self)
        self.h5_writer = h5_writer
        self.features = features
        self.files = files
    def run(self):
        for i, file in enumerate(self.files):
            image_id, _ = os.path.splitext(file)
            self.h5_writer.put(image_id, self.features[i])

if __name__ == "__main__":

    config = "configs/lsvrd-vgg19-512.json"
    batch_size = 32

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
    h5_writer = H5DataWriter(vcfg.cache_dir, cfg.dataset, len(files), 16, fields)

    print("building %s ..." % vcfg.backbone)
    backbone_cls = backbones[vcfg.backbone]
    cnn = backbone_cls()
    cnn.freeze()
    cnn.cuda()

    print("pre-extracting features with %s ..." % vcfg.backbone)
    n_batches = int(math.ceil(len(files)/batch_size))
    pbar = tqdm(total=n_batches*batch_size)

    batch_images = []
    loader = LoaderThread(files[:batch_size], vcfg.image_height, vcfg.image_width, batch_images)
    loader.start()

    writer = None

    for b in range(n_batches):

        loader.join()
        images = np.concatenate(batch_images, axis=0)
        if b + 1 < n_batches:
            batch_images = []
            loader = LoaderThread(files[(b+1)*batch_size: (b+2)*batch_size],
                                  vcfg.image_height, vcfg.image_width, batch_images)
            loader.start()

        images = torch.from_numpy(images).float().cuda()
        features = cnn(images).data.cpu().numpy()

        if writer is not None: writer.join()
        writer = WriterThread(h5_writer, features, files[b*batch_size: (b+1)*batch_size])
        writer.start()

        pbar.update(batch_size)

    pbar.close()
    writer.join()
    h5_writer.close()

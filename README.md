# Visual Relationship Encoder - PyTorch Implementation

## Environment

python==3.6 pytorch==0.4.0
anaconda recommended

install additional requirements with
```
pip install -r requirements.txt
```

clone roi_align (anywhere you like) 
```
git clone https://github.com/longcw/RoIAlign.pytorch.git
cd RoIAlign.pytorch
```
modify the `-arch` argument in `install.sh` to suite your GPU.
See `https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/` for more information
```
sh install.sh
sh test.sh # to see if it's correctly installed
```

## Data

download glove 6B word embeddings from `http://nlp.stanford.edu/data/glove.6B.zip`
unzip in a folder
make a soft link under `data/`

download the GQA dataset from `https://cs.stanford.edu/people/dorarad/gqa/download.html`
make a soft link under `data/`

make sure you have:
```
cd $VRE_ROOT
tree data/gqa -L 1
data/gqa
├── images - all the image files extracted in this folder
├── objects - object features (.h5 files and the .json)
├── questions12 - v1.2 question json files
└── scene_graphs - scene graph json files
tree data/glove -L 1
data/glove
└── glove.6B.300d.txt
```

preprocess data:
```
cd $LSVRD_ROOT
sh scripts/prepare_data.sh
```

## Training
Specify configuration file 
```
cd $LSVRD_ROOT
python main/train.py --config configs/lsvrd-res101-512.json
```
To see all the args
```
cd $LSVRD_ROOT
python main/train.py --help
```

## Structure
```
$LSVRD_ROOT
├── cache
├── configs # configuration files
├── data
│   ├── glove -> /path/to/glove/
│   └── gqa -> /path/to/gqa/
├── lib
│   ├── data
│   │   ├── dataset.py
│   │   ├── h5io.py
│   │   └── sym_dict.py
│   ├── evaluate.py # evaluation function
│   ├── loss
│   │   ├── consistency_loss.py
│   │   ├── triplet_loss.py
│   │   └── triplet_softmax_loss.py
│   ├── model
│   │   ├── language_model.py
│   │   ├── loss_model.py
│   │   └── vision_model.py
│   ├── module
│   │   ├── backbone.py
│   │   ├── entity_net.py
│   │   ├── relation_net.py
│   │   └── similarity_model.py
│   ├── train.py
│   └── utils.py
├── main
│   └── train.py
└── scripts
```

### Training

You can optionally use `scripts/pre_extract_features.py` to extract ResNet-101 feature maps as 
preprocessing.

example:
```
python main/train.py \
--config configs/resnet101-512-14-7-7-GRU-300d-1layer-5-0-256-128-0.2-0.2-1.0-1001-gt-311-100000-1e-4-0.8.json \
--n_epochs 5 \
```

run `python main/train.py --help` for other options.

### Inference

To extract features on GQA, use `main/infer.py`
example:
```
python main/train.py \
--config configs/
--lckpt path/to/language/model.pth
--vckpt path/to/vision/model.pth
--dataset gqa
```

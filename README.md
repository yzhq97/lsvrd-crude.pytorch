# Large-Scale Visual Relationship Understanding - PyTorch Implementation

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
cd $LSVRD_ROOT
tree data/gqa -L 1
data/gqa
├── images
├── objects
├── questions
└── scene_graphs
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





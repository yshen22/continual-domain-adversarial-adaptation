This repository contains the code accompanying the AISTATS2024 paper  "
Continual Domain Adversarial Adaptation via Double-Head Discriminators " Paper[link](https://arxiv.org/pdf/2402.03588.pdf): 

![network structure](algorithm_flow.png  "Problem description")

#### Requirements to run the code:
---

1. Python 3.8
2. Tensorflow 2.15.0
3. numpy 1.20.3
4. tqdm

#### Download dataset:
---

Download mnistm data:
```
curl -L -O http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz
```
Preprocess mnistm dataset
```
python create_mnistm.py 
```

#### Experiments on Continual Domain Adaptation:
---
Usage for supervised training on source domain at Phase S0:
```
python experiment.py -train_mod='sup_train' -SUP_EPOCHS=10 -adv_loss='MDD' -ckpt_path=$CHECKPOINT_SAVE_S0_DIR  
```

Usage for continual adversarial domain adaptation from the pretrained model at S0 using double-head domain discriminators: 
```
python experiment.py -SUP_EPOCHS=10 -SR_DISC_EPOCHS=5 -DA_EPOCHS=100 -adv_loss='MDD' -ckpt_path=$CHECKPOINT_SAVE_S0_DIR   
```

### Reference
---

```
@misc{2402.03588},
Author = {Yan Shen, Zhanghexuan Ji, Chunwei Ma, Mingchen Gao},
Title = {Continual Domain Adversarial Adaptation via Double-Head Discriminators},
Year = {2024},
Eprint = {arXiv:2402.03588},
}

This repository contains the code accompanying the AISTATS2024 paper  "
Continual Domain Adversarial Adaptation via Double-Head Discriminators " Paper[link](https://proceedings.mlr.press/v238/shen24a/shen24a.pdf): 

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

@InProceedings{pmlr-v238-shen24a,
  title = 	 { Continual Domain Adversarial Adaptation via Double-Head Discriminators },
  author =       {Shen, Yan and Ji, Zhanghexuan and Ma, Chunwei and Gao, Mingchen},
  booktitle = 	 {Proceedings of The 27th International Conference on Artificial Intelligence and Statistics},
  pages = 	 {2584--2592},
  year = 	 {2024},
  editor = 	 {Dasgupta, Sanjoy and Mandt, Stephan and Li, Yingzhen},
  volume = 	 {238},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {02--04 May},
  publisher =    {PMLR},
}

import tensorflow as tf
import numpy as np
import os
from functools import partial
from model import adv_loss
from model import backbone
from model import sup_loss
from train import sup
from train import da
from train import continual_da
from optimizer import optimizer_utils
import argparse
import tqdm
import importlib
import matplotlib
import pandas as pd
matplotlib.use('Agg') 
import dataset_factory
import json
import experiment_manager


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()
parser.add_argument('-adv_loss', type=str, default= 'MDD' )
# parser.add_argument('-base_net', type=str, default= 'mobileNet' )
parser.add_argument('-batch_size', type=int, default= 32)
parser.add_argument('-dataset', type=str, default='mnist')
parser.add_argument('-max_iter', type=int , default =5000 )
parser.add_argument('-gpu', type=str , default ='1' )
parser.add_argument('-dataset_source', type=str, default ='amazon')
parser.add_argument('-dataset_target', type=str, default='dslr')
parser.add_argument('-l2_decay', type=float, default= 0.0001)
parser.add_argument('-use_l2', type=str2bool, default= False)
parser.add_argument('-num_class', type=int, default = 10)
parser.add_argument('-num_hidden1', type=int, default = 100)
parser.add_argument('-result_dir', type=str, default='exp_result')
parser.add_argument('-num_sample_per_class', type=int, default=10)
parser.add_argument('-train_mod', type=str, default= 'continual_da')
parser.add_argument('-alpha1', type=float, default=0.1)
parser.add_argument('-alpha2', type=float, default=1.)
parser.add_argument('-gamma', type=float, default=1.)
parser.add_argument('-beta', type=float, default=0.15)
parser.add_argument('-SUP_EPOCHS', type=int, default=15)
parser.add_argument('-DA_EPOCHS', type=int, default=100)
parser.add_argument('-SR_DISC_EPOCHS', type=int, default=5)
parser.add_argument('-td_optimizer', type=str, default='adam')
parser.add_argument('-sd_optimizer', type=str, default='adam')
parser.add_argument('-f_optimizer', type=str, default='adam')
parser.add_argument('-td_scheduler', type=str, default='constant')
parser.add_argument('-sd_scheduler', type=str, default='constant')
parser.add_argument('-f_scheduler', type=str, default='constant')
parser.add_argument('-td_lr', type=float, default=0.001)
parser.add_argument('-sd_lr', type=float, default=0.0001)
parser.add_argument('-f_lr', type=float, default=0.001)
parser.add_argument('-f_lr_gamma', default=0.0002, type=float)
parser.add_argument('-f_lr_decay', default=0.75, type=float)
parser.add_argument('-f_momentum', default=0.9, type=float)
parser.add_argument('-sd_momentum', default=0., type=float)
parser.add_argument('-td_momentum', default=0., type=float)
parser.add_argument('-use_source_disc', type=str2bool, default=True)
parser.add_argument('-ckpt_path', type=str, default='checkpoint/mnistm')

args=parser.parse_args()
print(args)

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu

def load_checkpoint(ckpt_path):
    for image, label in source_train_dataset:
        test_step(image, label)
        break
    restorer = tf.train.Checkpoint()
    restorer.mapped = {'feature_generator': feature_generator.variables}
    restorer.mapped.update({'label_predictor': label_predictor.variables})
    restorer.restore(ckpt_path)

print(tf.config.list_physical_devices('GPU'))

if __name__ == '__main__':
    dataset_gen = dataset_factory.load_data_iterator(args.dataset)
    source_train_dataset, source_test_dataset, target_train_dataset, target_test_dataset, mem_dataset = dataset_gen(args.dataset_source, args.dataset_target, args.num_sample_per_class
    , args.batch_size)
    FeatureGenerator = backbone.get_feature_generator(args.dataset) 
    LabelPredictor = backbone.get_label_predictor(args.dataset)
    feature_generator = FeatureGenerator()
    label_predictor = LabelPredictor(use_activation=True)
    if args.adv_loss == 'MDD':
       domain_predictor_source = LabelPredictor()
       domain_predictor_target = LabelPredictor()
    test_accuracy = sup_loss.get_test_accuracy()
#    td_optimizer = optimizer_utils.get_optimizer(vars(args), 'td')
#     f_lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
#         0.001,
#         decay_steps=50000,
#         decay_rate=0.95,
#         staircase=True)
    # td_lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    #     0.001,
    #     decay_steps=100000,
    #     decay_rate=0.96,
    #     staircase=True)
    td_optimizer = tf.keras.optimizers.Adam(0.001)
    sd_optimizer = optimizer_utils.get_optimizer(vars(args), 'sd')
    f_optimizer = tf.keras.optimizers.Adam(0.001)
#    f_optimizer = optimizer_utils.get_optimizer(vars(args), 'f')
#    print(f_optimizer)
    metrics_mngr = experiment_manager._setup_outputs(args.result_dir, vars(args))
    @tf.function
    def sup_train_step(image, label):
       sup_loss.sup_train_step(image, label, feature_generator, label_predictor, f_optimizer, args.alpha1, args.alpha2)
    @tf.function
    def da_train_step(tar_image, sr_image, sr_label):
      adv_loss.mdd_semi_sup_step(tar_image, sr_image, sr_label, feature_generator, label_predictor, domain_predictor_source
      , domain_predictor_target, sd_optimizer, td_optimizer, f_optimizer, args.beta, args.alpha1, args.alpha2, args.gamma, use_source_disc=args.use_source_disc)
    @tf.function
    def disc_source_only_step(sr_image):
      adv_loss.mdd_disc_source_only_step(sr_image, feature_generator, label_predictor, domain_predictor_source, sd_optimizer)
    @tf.function
    def test_step(image, label):
      sup_loss.test_step(image, label, feature_generator, label_predictor, test_accuracy)
    if args.train_mod == 'sup_train':
      sup.train(args.SUP_EPOCHS ,source_train_dataset, source_test_dataset, target_test_dataset, sup_train_step, test_step, test_accuracy, metrics_mngr)
      saver = tf.train.Checkpoint()
      saver.mapped = {'feature_generator': feature_generator.variables}
      saver.mapped.update({'label_predictor': label_predictor.variables})
      saver.save(args.ckpt_path)

    elif args.train_mod == 'da_train':
      da.train(args.DA_EPOCHS, source_train_dataset, source_test_dataset, target_train_dataset, target_test_dataset, 
      da_train_step, test_step, test_accuracy, metrics_mngr)
    elif args.train_mod == 'continual_da':
      load_checkpoint(args.ckpt_path)
      continual_da.train(args.SUP_EPOCHS, args.SR_DISC_EPOCHS, args.DA_EPOCHS, source_train_dataset, source_test_dataset, mem_dataset, target_train_dataset
      , target_test_dataset, sup_train_step, disc_source_only_step, da_train_step, test_step
      , test_accuracy, metrics_mngr)
    

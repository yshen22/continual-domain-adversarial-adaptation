import tensorflow as tf
from functools import partial

_SUPPORTED_OPTIMIZERS_CLS = [
    tf.keras.optimizers.SGD,
    tf.keras.optimizers.Adagrad,
    tf.keras.optimizers.Adam,
]

_SUPPORTED_OPTIMIZERS = {
    cls.__name__.lower(): cls for cls in _SUPPORTED_OPTIMIZERS_CLS
}


# def exp_decay_scheduler(steps, lr0, lr_gamma, lr_decay):
# 	return lr0 * (1. + lr_gamma * float(step)) ** (lr_decay)

class exp_decaySchedule(tf.keras.optimizers.schedules.LearningRateSchedule):

	def __init__(self, lr0, lr_gamma, lr_decay):
		self.lr0 = lr0
		self.lr_gamma = lr_gamma
		self.lr_decay = lr_decay
	def __call__(self, step):
		return self.lr0 * (1. + self.lr_gamma * step) ** (self.lr_decay)

def get_optimizer(arg_dict, mod):
	for key in arg_dict.keys():
		if (key.split('_')[0] == mod and key.split('_')[1] == 'optimizer'):
			optimizer_key = key
			break 
	optimizers = _SUPPORTED_OPTIMIZERS[arg_dict[optimizer_key]]
	lr0 = arg_dict[mod +'_' + 'lr']
	if arg_dict[optimizer_key] == 'sgd' and arg_dict[mod + '_' + 'momentum'] > 1e-4:
		optimizers = partial(optimizers, momentum = arg_dict[mod + '_' + 'momentum'])
	for key in arg_dict.keys():
		if (key.split('_')[0] == mod and key.split('_')[1] == 'scheduler'):
			schedule_key = key
			break
	# if arg_dict['layerlr'] and mod == 'f':
	# 	lr1 = lr0 * arg_dict['alpha1']
	# 	if arg_dict[schedule_key] == 'constant':
	# 		if arg_dict[optimizer_key] == 'adamw':
	# 			return (optimizers(weight_decay=1e-7, learning_rate=lr1), optimizers(weight_decay=1e-7, learning_rate=lr0))
	# 		else :
	# 			return (optimizers(learning_rate=lr1), optimizers(learning_rate=lr0), optimizers(learning_rate=lr0))
	# 	elif arg_dict[schedule_key] == 'exp_decay':
	# 		return (optimizers(learning_rate = exp_decaySchedule(lr0=lr1, lr_gamma=arg_dict[mod +'_' + 'lr_gamma'], lr_decay=arg_dict[mod +'_' + 'lr_decay'])),
	# 				optimizers(learning_rate=exp_decaySchedule(lr0=lr0, lr_gamma=arg_dict[mod + '_' + 'lr_gamma'],
	# 														   lr_decay=arg_dict[mod + '_' + 'lr_decay'])))
	# else :
	if arg_dict[schedule_key] == 'constant':
		if arg_dict[optimizer_key] == 'adamw':
			return optimizers(weight_decay=1e-7, learning_rate=lr0)
		else :
			return optimizers(learning_rate=arg_dict[mod +'_' + 'lr'])
	elif arg_dict[schedule_key] == 'exp_decay':
		return optimizers(learning_rate = exp_decaySchedule(lr0=lr0, lr_gamma=arg_dict[mod +'_' + 'lr_gamma'], lr_decay=arg_dict[mod +'_' + 'lr_decay']))
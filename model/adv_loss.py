import tensorflow as tf
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
def mdd_semi_sup_step(tar_image, mem_image, mem_label, feature_generator, label_predictor, domain_predictor_source
      , domain_predictor_target, sd_optimizer, td_optimizer, f_optimizer, beta, alpha1, alpha2, gamma, use_source_disc= True):
   # print(tar_image.get_shape().as_list())
   # print(mem_image.get_shape().as_list())
   with tf.GradientTape(persistent=True) as tape:
      t_features = feature_generator(tar_image, is_train=False)
      s_features = feature_generator(mem_image, is_train=False)
      sd_predictions = domain_predictor_target(s_features, is_train=True)
      td_predictions = domain_predictor_target(t_features, is_train=True)
      pesudo_label_s = tf.cast(tf.argmax(label_predictor(s_features, is_train=False), axis=1), tf.int32)
      pesudo_label_t = tf.cast(tf.argmax(label_predictor(t_features, is_train=False), axis=1), tf.int32)   
      cat_idx = tf.stack([tf.range(0, tf.shape(pesudo_label_t)[0]), pesudo_label_t], axis=1)    
      domain_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits= sd_predictions, labels= pesudo_label_s)
      - tf.gather_nd(tf.math.log(1.00001 - tf.nn.softmax(logits = td_predictions, axis=1)), cat_idx))
      total_loss = domain_loss
   d_gradients_on_loss = tape.gradient(total_loss, domain_predictor_target.trainable_variables)
   td_optimizer.apply_gradients(zip(d_gradients_on_loss, domain_predictor_target.trainable_variables))
   
   with tf.GradientTape(persistent=True) as tape:
      t_features = feature_generator(tar_image, is_train=True)
      l_predictions = label_predictor(feature_generator(mem_image, is_train=True), is_train=True)
      if use_source_disc :
         d_predictions = domain_predictor_target(t_features, is_train=False) + gamma*domain_predictor_source(t_features, is_train=False)
      else :
         d_predictions = domain_predictor_target(t_features, is_train=False)
#      pesudo_label_t = tf.cast(tf.argmax(label_predictor(t_features, is_train=False), axis=1), tf.int32)
      cat_idx = tf.stack([tf.range(0, tf.shape(pesudo_label_t)[0]), pesudo_label_t], axis=1)
      label_loss = loss_object(mem_label, l_predictions)
#      label_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits= l_predictions, labels=mem_label))
      domain_loss = tf.reduce_mean(-tf.gather_nd(tf.math.log(1.00001 - tf.nn.softmax(logits = d_predictions, axis=1)), cat_idx))
      total_loss = label_loss - beta * domain_loss
   l_gradients = tape.gradient(label_loss, label_predictor.trainable_variables)
   
   f_gradients =  tape.gradient(total_loss, feature_generator.trainable_variables)
   # bb_gradient_on_total_loss = tape.gradient(total_loss, feature_generator.backbone.trainable_variables)
   # bn_gradient_on_total_loss = tape.gradient(total_loss, feature_generator.bottleneck.trainable_variables)
   # bb_gradient_on_total_loss = [alpha1 * bb_gradient_on_total_loss[i] for i in range(len(bb_gradient_on_total_loss))]
   # bn_gradient_on_total_loss = [alpha2 * bn_gradient_on_total_loss[i] for i in range(len(bn_gradient_on_total_loss))]
   f_optimizer.apply_gradients(zip(f_gradients + l_gradients,
     feature_generator.trainable_variables + label_predictor.trainable_variables))
   # f_optimizer.apply_gradients(zip(bb_gradient_on_total_loss + bn_gradient_on_total_loss + l_gradients,
   #            feature_generator.backbone.trainable_variables + feature_generator.bottleneck.trainable_variables + label_predictor.trainable_variables))

def mdd_disc_source_only_step(src_image, feature_generator, label_predictor, domain_predictor_source, sd_optimizer):
   with tf.GradientTape(persistent=True) as d_tape:
         features = feature_generator(src_image, is_train=False)
         pesudo_label = tf.cast(tf.argmax(label_predictor(features, is_train=False), axis=1), tf.int32)
         d_predictions_all = domain_predictor_source(features, is_train=True)
         # with tf.GradientTape(persistent=True) as f_tape:
         #    features = feature_generator(src_image, is_train=False)
         #    pesudo_label = tf.cast(tf.argmax(label_predictor(features, is_train=False), axis=1), tf.int32)
         #    d_predictions_all = domain_predictor_source(features, is_train=True)
         #    cat_idx = tf.stack([tf.range(0, tf.shape(pesudo_label)[0]), pesudo_label], axis=1)  
         #    d_predictions = tf.gather_nd(d_predictions_all, cat_idx)
         # grads = f_tape.gradient(d_predictions, features)
         # grad_norms = tf.reduce_sum(tf.square(grads), axis=1)**6
         # grad_penalty = tf.reduce_mean(grad_norms)
         domain_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits= d_predictions_all, labels= pesudo_label)
         total_loss = domain_loss 
   d_gradients = d_tape.gradient(total_loss, domain_predictor_source.trainable_variables)
   sd_optimizer.apply_gradients(zip(d_gradients, domain_predictor_source.trainable_variables))
        

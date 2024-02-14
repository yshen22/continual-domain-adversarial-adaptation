import tensorflow as tf
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
def sup_train_step(images, labels, feature_generator, label_predictor, f_optimizer, alpha1, alpha2):
    with tf.GradientTape(persistent=True) as tape:
        new_features = feature_generator(images, is_train=True)
        l_predictions = label_predictor(new_features, is_train=True)
        label_loss = loss_object(labels, l_predictions)
#        label_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=l_predictions, labels= labels))
    # bb_gradient_on_label = tape.gradient(label_loss, feature_generator.backbone.trainable_variables)
    # bn_gradient_on_label = tape.gradient(label_loss, feature_generator.bottleneck.trainable_variables)
    f_gradient_on_label =  tape.gradient(label_loss, feature_generator.trainable_variables)
    l_gradient_on_label =  tape.gradient(label_loss, label_predictor.trainable_variables)
    # bb_gradient_on_label = [alpha1 * bb_gradient_on_label[i] for i in range(len(bb_gradient_on_label))]
    # bn_gradient_on_label = [alpha2 * bn_gradient_on_label[i] for i in range(len(bn_gradient_on_label))]
    f_optimizer.apply_gradients(zip(f_gradient_on_label + l_gradient_on_label,
     feature_generator.trainable_variables + label_predictor.trainable_variables))
    # f_optimizer.apply_gradients(zip(bb_gradient_on_label + bn_gradient_on_label + l_gradient_on_label,
    #  feature_generator.backbone.trainable_variables+feature_generator.bottleneck.trainable_variables +label_predictor.trainable_variables))
    return

@tf.function
def test_step(images, labels, feature_generator, label_predictor, test_accuracy):
    features = feature_generator(images, is_train=False)
    predictions = label_predictor(features, is_train=False)
    test_accuracy(labels, predictions)

def get_test_accuracy():
    return tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

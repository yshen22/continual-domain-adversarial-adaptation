def train(sup_epochs, sr_disc_epochs, da_epochs, source_train_ds, source_test_ds, mem_ds, target_train_ds, target_test_ds, sup_train_step
, sr_disc_train_step, da_train_step, test_step, test_accuracy, metrics_mngr):
    result_dict = {'source domain acc': 0. , 'target domain acc': 0.}
    # for epoch in range(sup_epochs):
    #     test_accuracy.reset_states()
    #     for images, labels in source_train_ds:
    #       sup_train_step(images, labels)
    #     for test_data in source_train_ds:
    #       test_step(test_data[0], test_data[1])
    #     template = 'Epoch {}, Source Test Accuracy: {}'
    #     print (template.format(epoch+1,
    #                      test_accuracy.result()*100,
    #                      ))
    #     result_dict['source domain acc'] = test_accuracy.result().numpy()*100 
    #     test_accuracy.reset_states()
    #     for test_data in target_test_ds:
    #       test_step(test_data[0], test_data[1])
    #     template = 'Epoch {}, Target Test Accuracy: {}'
    #     print (template.format(epoch+1,
    #                      test_accuracy.result()*100,
    #                      ))
    #     result_dict['target domain acc'] = test_accuracy.result().numpy()*100
    #     metrics_mngr.update_metrics(epoch+1, result_dict)
    test_accuracy.reset_states()
    for epoch in range(sr_disc_epochs):
        for images, labels in source_train_ds:
            sr_disc_train_step(images)
    for test_data in source_test_ds:
        test_step(test_data[0], test_data[1])
    template = 'Epoch {}, Source Test Accuracy: {}'
    print (template.format(epoch+1,
                     test_accuracy.result()*100,
                     ))
    result_dict['source domain acc'] = test_accuracy.result().numpy()*100 
    test_accuracy.reset_states()
    for test_data in target_test_ds:
      test_step(test_data[0], test_data[1])
    template = 'Epoch {}, Target Test Accuracy: {}'
    print (template.format(epoch+1,
                     test_accuracy.result()*100,
                     ))
    result_dict['target domain acc'] = test_accuracy.result().numpy()*100
    metrics_mngr.update_metrics(1, result_dict)

    for epoch in range(da_epochs):
        test_accuracy.reset_states()
        for (sr_images, sr_labels), (ta_images, ta_labels) in zip(mem_ds, target_train_ds):
            da_train_step(ta_images, sr_images, sr_labels)
        for test_data in source_test_ds:
            test_step(test_data[0], test_data[1])
        template = 'Epoch {}, Source Test Accuracy: {}'
        print (template.format(epoch+1,
                         test_accuracy.result()*100,
                         ))
        result_dict['source domain acc'] = test_accuracy.result().numpy()*100 
        test_accuracy.reset_states()
        for test_data in target_test_ds:
          test_step(test_data[0], test_data[1])
        template = 'Epoch {}, Target Test Accuracy: {}'
        print (template.format(epoch+1,
                         test_accuracy.result()*100,
                         ))
        result_dict['target domain acc'] = test_accuracy.result().numpy()*100
        metrics_mngr.update_metrics(sup_epochs+ epoch+1, result_dict)

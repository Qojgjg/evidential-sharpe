python main.py \
    feature_config.normalization_scheme=raw \
    model_config=bin_tabl \
    loss_config=EDLLoss \
    dataloader_config.batch_size=512 \
    train_config.n_epoch=200
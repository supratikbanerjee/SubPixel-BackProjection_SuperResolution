def create_model(config, logger):
    model = config['model']
    
    if model == 'cnn':
        from .cnn_trainer import CNN as M

    else:
        raise NotImplementedError('Model [{:s}] not recognized.'.format(model))
    m = M(config)
    logger.log('Model [{:s}] is created.'.format(m.__class__.__name__))
    return m

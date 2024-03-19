def create_model(opt):
    model = None
    print(opt.model)
    if opt.model == 'resvit_one':
        from .resvit_one import ResViT_model
        model = ResViT_model()
    else:
        raise NotImplementedError('model [%s] not implemented.' % opt.model)
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    return model

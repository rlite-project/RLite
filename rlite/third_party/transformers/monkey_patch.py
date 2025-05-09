import transformers


def patch_from_pretrained(cls):
    orig_from_pretrained = cls.from_pretrained

    def new_from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        config = orig_from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        config.name_or_path = pretrained_model_name_or_path
        return config

    cls.from_pretrained = classmethod(new_from_pretrained)


def patch_all_configs():
    for subclass in transformers.PretrainedConfig.__subclasses__():
        patch_from_pretrained(subclass)


patch_all_configs()

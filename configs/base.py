class Config(dict):

    def __getattr__(self, k):
        if k not in self:
            raise AttributeError(f'{k} is not in config.')
        return self[k]
    
    def __setattr__(self, k, v):
        self[k] = v
    
    def __delattr__(self, k):
        del self[k]
    
    @staticmethod
    def copy_from_dict(d):
        if isinstance(d, dict):
            subconfig = Config()
            for k in d:
                subconfig[k] = Config.copy_from_dict(d[k])
            return subconfig
        else:
            return d


if __name__ == '__main__':
    import pdb; pdb.set_trace()
    config = Config.copy_from_dict({
        'exp_name': 'test',
        'data_config': {
            'data_dir': 'test_dir'
        }
    })
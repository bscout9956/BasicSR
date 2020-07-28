import os
import os.path as osp
import logging
import re


def parse(opt_path, is_train=True):
    extension = osp.splitext(opt_path)[1].lower()
    if extension == '.json':
        import json
        # remove comments starting with '//'
        json_str = ''
        with open(opt_path, 'r') as f:
            for line in f:
                line = line.split('//')[0] + '\n'
                json_str += line
        opt = json.loads(json_str)
    elif extension == '.cson':
        import cson
        with open(opt_path, 'r') as f:
            opt = cson.load(f)
    elif extension == '.yml' or extension == '.yaml':
        import yaml
        loader = yaml.SafeLoader
        # fix parsing of floats
        loader.add_implicit_resolver(
            u'tag:yaml.org,2002:float',
            re.compile(u'''^(?:
             [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
            |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
            |\\.[0-9_]+(?:[eE][-+]?[0-9]+)?
            |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
            |[-+]?\\.(?:inf|Inf|INF)
            |\\.(?:nan|NaN|NAN))$''', re.X),
            list(u'-+0123456789.'))
        with open(opt_path, 'r') as f:
            opt = yaml.load(f, Loader=loader)
    else:
        raise ValueError('Unknown file extension: {}'.format(extension))

    opt['is_train'] = is_train
    scale = opt['scale']

    # datasets
    for phase, dataset in opt['datasets'].items():
        phase = phase.split('_')[0]
        dataset['phase'] = phase
        dataset['scale'] = scale
        is_lmdb = False
        if 'dataroot_HR' in dataset and dataset['dataroot_HR'] is not None:
            HR_images_paths = dataset['dataroot_HR']        
            if type(HR_images_paths) is list:
                dataset['dataroot_HR'] = []
                for path in HR_images_paths:
                    dataset['dataroot_HR'].append(os.path.expanduser(path))
                    # if dataset['dataroot_HR'].endswith('lmdb'): #missing, how to check for lmdb with a list?
                        # is_lmdb = True
            elif type(HR_images_paths) is str:
                dataset['dataroot_HR'] = os.path.expanduser(HR_images_paths)
                if dataset['dataroot_HR'].endswith('lmdb'):
                    is_lmdb = True
        if 'dataroot_HR_bg' in dataset and dataset['dataroot_HR_bg'] is not None:
            HR_images_paths = dataset['dataroot_HR_bg']        
            if type(HR_images_paths) is list:
                dataset['dataroot_HR_bg'] = []
                for path in HR_images_paths:
                    dataset['dataroot_HR_bg'].append(os.path.expanduser(path))
            elif type(HR_images_paths) is str:
                dataset['dataroot_HR_bg'] = os.path.expanduser(HR_images_paths)
        if 'dataroot_LR' in dataset and dataset['dataroot_LR'] is not None:
            LR_images_paths = dataset['dataroot_LR']        
            if type(LR_images_paths) is list:
                dataset['dataroot_LR'] = []
                for path in LR_images_paths:
                    dataset['dataroot_LR'].append(os.path.expanduser(path))
                    # if dataset['dataroot_HR'].endswith('lmdb'): #missing, how to check for lmdb with a list?
                        # is_lmdb = True
            elif type(LR_images_paths) is str:
                dataset['dataroot_LR'] = os.path.expanduser(LR_images_paths)
                if dataset['dataroot_LR'].endswith('lmdb'):
                    is_lmdb = True
        dataset['data_type'] = 'lmdb' if is_lmdb else 'img'
        dataset['LR_nc'] = opt['network_G']['in_nc']
        dataset['HR_nc'] = opt['network_G']['out_nc']

        if phase == 'train' and 'subset_file' in dataset and dataset['subset_file'] is not None:
            dataset['subset_file'] = os.path.expanduser(dataset['subset_file'])

    # path
    for key, path in opt['path'].items():
        if path and key in opt['path']:
            opt['path'][key] = os.path.expanduser(path)
    if is_train:
        experiments_root = os.path.join(opt['path']['root'], 'experiments', opt['name'])
        opt['path']['experiments_root'] = experiments_root
        opt['path']['models'] = os.path.join(experiments_root, 'models')
        opt['path']['training_state'] = os.path.join(experiments_root, 'training_state')
        opt['path']['log'] = experiments_root
        opt['path']['val_images'] = os.path.join(experiments_root, 'val_images')

        # change some options for debug mode
        if 'debug' in opt['name']:
            opt['train']['val_freq'] = 8
            opt['logger']['print_freq'] = 2
            opt['logger']['save_checkpoint_freq'] = 8
            opt['logger']['backup_freq'] = 2
            opt['train']['lr_decay_iter'] = 10
    else:  # test
        results_root = os.path.join(opt['path']['root'], 'results', opt['name'])
        opt['path']['results_root'] = results_root
        opt['path']['log'] = results_root

    # network
    opt['network_G']['scale'] = scale

    # batch multiplier
    if not 'batch_multiplier' in opt or opt['batch_multiplier'] is None:
        opt['batch_multiplier'] = 1

    # export CUDA_VISIBLE_DEVICES
    gpu_list = ','.join(str(x) for x in opt['gpu_ids'])
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
    print('export CUDA_VISIBLE_DEVICES=' + gpu_list)

    return opt


class NoneDict(dict):
    def __missing__(self, key):
        return None


# convert to NoneDict, which return None for missing key.
def dict_to_nonedict(opt):
    if isinstance(opt, dict):
        new_opt = dict()
        for key, sub_opt in opt.items():
            new_opt[key] = dict_to_nonedict(sub_opt)
        return NoneDict(**new_opt)
    elif isinstance(opt, list):
        return [dict_to_nonedict(sub_opt) for sub_opt in opt]
    else:
        return opt


def dict2str(opt, indent_l=1):
    '''dict to string for logger'''
    msg = ''
    for k, v in opt.items():
        if isinstance(v, dict):
            msg += ' ' * (indent_l * 2) + k + ':[\n'
            msg += dict2str(v, indent_l + 1)
            msg += ' ' * (indent_l * 2) + ']\n'
        else:
            msg += ' ' * (indent_l * 2) + k + ': ' + str(v) + '\n'
    return msg


def check_resume(opt):
    '''Check resume states and pretrain_model paths'''
    logger = logging.getLogger('base')
    if opt['path']['resume_state']:
        if opt['path']['pretrain_model_G'] or opt['path']['pretrain_model_D']:
            logger.warning('pretrain_model path will be ignored when resuming training.')

        if 'backup.state' in opt['path']['resume_state']:
            name = 'backup'
        else:
            state_idx = osp.basename(opt['path']['resume_state']).split('.')[0]
            name = '{}_{}'.format(opt['name'], state_idx)
                
        opt['path']['pretrain_model_G'] = osp.join(opt['path']['models'],'{}_G.pth'.format(name))
        logger.info('Set [pretrain_model_G] to ' + opt['path']['pretrain_model_G'])
        if 'gan' in opt['model']:
            opt['path']['pretrain_model_D'] = osp.join(opt['path']['models'], '{}_D.pth'.format(name))
            logger.info('Set [pretrain_model_D] to ' + opt['path']['pretrain_model_D'])
    
    

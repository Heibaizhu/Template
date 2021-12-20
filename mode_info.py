from models.archs import define_network
import yaml
import os
from ptflops import get_model_complexity_info
from scripts.WatchReceptiveField import GetERF
import torch
from copy import deepcopy
from torch.nn.parallel import DataParallel, DistributedDataParallel


def load_network(net, load_path, strict=True, param_key='params'):
    """Load network.

    Args:
        load_path (str): The path of networks to be loaded.
        net (nn.Module): Network.
        strict (bool): Whether strictly loaded.
        param_key (str): The parameter key of loaded network. If set to
            None, use the root 'path'.
            Default: 'params'.
    """
    def get_bare_model(net):
        """Get bare model, especially under wrapping with
        DistributedDataParallel or DataParallel.
        """
        # TODO: check if net.module exists
        if isinstance(net, (DataParallel, DistributedDataParallel)):
            net = net.module
        return net
    net = get_bare_model(net)
    load_net = torch.load(
        load_path, map_location=lambda storage, loc: storage)
    if param_key is not None:
        load_net = load_net[param_key]
    # remove unnecessary 'module.'
    for k, v in deepcopy(load_net).items():
        if k.startswith('module.'):
            load_net[k[7:]] = v
            load_net.pop(k)
    net.load_state_dict(load_net, strict=strict)
    return net

if __name__ == '__main__':
    yaml_path = os.path.join('options', 'info', 'model_config.yml')
    with open(yaml_path, 'r') as f:
        opt = yaml.load(f, Loader=yaml.FullLoader)
    state = opt['network_g'].pop('resume_state')
    input_shape = opt['network_g'].pop('input_shape')
    net = define_network(opt['network_g'])

    if state:
        net = load_network(net, state)

    # 计算模型的大小及其复杂度
    macs, params = get_model_complexity_info(net, tuple(input_shape[1:]), as_strings=True,
                                             print_per_layer_stat=True, verbose=True)
    print('{:<30}  {:<8}'.format('Computational floaps: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))

    # from torchsummary import summary

    # summary(net.to('cuda'), (3, 256, 256))

    #计算感受野
    if state:
        iters = opt['iters']
        centerCrop = opt['centerCrop']
        savePath = opt['savePath']
        obj = GetERF()
        grads = obj(net, input_shape, iters=iters)
        obj.save(grads, outPath=savePath, centerCrop=centerCrop)

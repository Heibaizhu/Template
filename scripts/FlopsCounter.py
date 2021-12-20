from ptflops import get_model_complexity_info


if __name__ == '__main__':
    import torchvision.models as models
    import torch

    with torch.cuda.device(0):
        net = models.densenet121()
        macs, params = get_model_complexity_info(net, (3, 224, 224), as_strings=True,
                                                 print_per_layer_stat=True, verbose=True)
        # print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))
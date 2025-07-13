import os
from model.model_resnet_s import *
from model.model_t2tvit_p_s import *
from model.model_resnetlarge_s import *
from model.model_t2tvit_t_s import *

from model.model_cnn_s import *
from model.networks import ResNet18, ResNet50

def get_default_convnet_setting():
    net_width, net_depth, net_act, net_norm, net_pooling = 128, 3, 'relu', 'instancenorm', 'avgpooling'
    return net_width, net_depth, net_act, net_norm, net_pooling


def get_model(model_name, exp, data_storage, num_classes, device, load=True):
    torch.random.manual_seed(int(time.time() * 1000) % 100000)
    net_width, net_depth, net_act, net_norm, net_pooling = get_default_convnet_setting()


    
    if 'resnet_s' in model_name:
        # net= ResNet18s(num_classes=num_classes).to(device)
        net = ResNet18(channel=3, num_classes= num_classes).to(device)
        file_path = os.path.join(data_storage, f'pretrained_{model_name}_exp_{exp}.pth')
        if load:
            net.load_state_dict(torch.load(file_path))


    elif 'cnn_s' in model_name:
        net = ConvNet(channel=3, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling, im_size=(32,32)).to(device)
        file_path = os.path.join(data_storage, f'pretrained_{model_name}_exp_{exp}.pth')
        if load:
            net.load_state_dict(torch.load(file_path))



    elif 'vit_s' in model_name:
        net= T2TVisionTransformerP_S(num_classes=num_classes).to(device)
        file_path = os.path.join(data_storage, f'pretrained_{model_name}_exp_{exp}.pth')
        if load:
            net.load_state_dict(torch.load(file_path))


    elif 'vitt_s' in model_name:
        net= T2TVisionTransformerT_S(num_classes=num_classes).to(device)
        file_path = os.path.join(data_storage, f'pretrained_{model_name}_exp_{exp}.pth')
        if load:
            net.load_state_dict(torch.load(file_path))


    elif 'resnetlarge_s' in model_name:
        net = ResNet50(channel=3, num_classes= num_classes).to(device)
        file_path = os.path.join(data_storage, f'pretrained_{model_name}_exp_{exp}.pth')
        if load:
            net.load_state_dict(torch.load(file_path))


    else:
        raise NotImplementedError


    return net


'''
    implement of SSRNet mxnet version
    (SSR-Net: A Compact Soft Stagewise Regression Network for Age Estimation)
    MXNet(静态版、动态版), 推荐 mxnet.gluon
    2D format is [N,C,H,W] !!!   Not [N,H,W,C]
    author is leilei
'''
##############################################################################
'''
    静态版 Symbolic API 符号式
'''
import mxnet as mx

def conv_bn_act(data, num_filter, activation, name, pool_type=None):
    '''
    :param data:        Symbol, input symbol type [N,C,H,W]
    :param num_filter:  int, output_channels
    :param activation:  str, activation functions ['relu', 'sigmoid', 'softrelu', 'softsign', 'tanh']
    :param name:        str, layer's name(Unique)
    :param pool_type:   str, pool's type ['max', 'avg'...], default=None
    :return:            conv2d batch_norm activation pool function
    '''
    conv = mx.sym.Convolution(data=data, kernel=(3,3), stride=(1,1), num_filter=num_filter, name=name+'_conv')  # no padding and has bias
    bn = mx.sym.BatchNorm(data=conv, fix_gamma=False, eps=2e-5, momentum=0.9, axis=1, name=name+'_bn')  # pytorch tf keras不需要设置什么参数，默认即可， mxnet则不行
    out = mx.sym.Activation(data=bn, act_type=activation, name=name+'_act')
    if pool_type is not None:
        out = mx.sym.Pooling(out, kernel=(2,2), stride=(2,2), pool_type=pool_type, name=name+'_pool')
    return out

def conv_act_drop_fully(data, num_filter, Kth_stage_num, name, pool_type='avg', pool_size=None):
    '''
    :param data:         Symbol
    :param num_filter:   int, output channels
    # :param activation:   str, activation functions ['relu', 'sigmoid', 'softrelu', 'softsign', 'tanh']
    :param Kth_stage_num:int,
    :param name:         str, layer's name
    :param pool_type:    str,
    :param pool_size:    int_tuple
    :return:             2 次结果
    '''
    conv = mx.sym.Convolution(data=data, kernel=(1,1), stride=(1,1), num_filter=num_filter, name=name+'_conv')
    act = mx.sym.Activation(data=conv, act_type='relu', name=name+'_act')
    if pool_size is not None:
        act = mx.sym.Pooling(data=act, kernel=pool_size, pool_type=pool_type, name=name+'_pool')
    layer = mx.sym.Flatten(data=act, name=name+'_fla')

    drop = mx.sym.Dropout(data=layer, p=0.2, name=name+'_mix_drop')
    fully = mx.sym.FullyConnected(data=drop, num_hidden=Kth_stage_num, name=name+'_mix_fully')
    layer_mix = mx.sym.Activation(data=fully, act_type='relu', name=name+'_mix_act')

    return layer, layer_mix

def fully_act(data, activation, Kth_stage_num, name):
    '''
    :param data:         Symbol
    :param activation:   str, activation functions
    :param Kth_stage_num:int
    :param name:         str, layer's name
    :return:
    '''
    fully = mx.sym.FullyConnected(data=data, num_hidden=Kth_stage_num, name=name+'_fully')
    out = mx.sym.Activation(data=fully, act_type=activation, name=name+'_act')
    return out

def fusion_block_and_regression(x_layer, s_layer, Kth_stage_num, num_filter, name, pool_size=None):
    '''
    :param x_layer:        Symbol stream 1
    :param s_layer:        Symbol stream 2
    :param Kth_stage_num:
    :param num_filter:     int, 10
    :param name:
    :return:  3个结果
    '''
    x_layer, x_layer_mix = conv_act_drop_fully(data=x_layer, num_filter=num_filter, Kth_stage_num=Kth_stage_num, name=name+'_x', pool_type='avg', pool_size=pool_size)
    s_layer, s_layer_mix = conv_act_drop_fully(data=s_layer, num_filter=num_filter, Kth_stage_num=Kth_stage_num, name=name+'_s', pool_type='max', pool_size=pool_size)

    feat_pre = s_layer * x_layer
    delta_s = fully_act(data=feat_pre, activation='tanh', Kth_stage_num=1, name=name+'_delta')  # dynamic width

    feat_pre = s_layer_mix * x_layer_mix
    pred1 = fully_act(data=feat_pre, activation='relu', Kth_stage_num=2*Kth_stage_num, name=name+'_pred1')  #-----------|
    pred_s = fully_act(data=pred1, activation='relu', Kth_stage_num=Kth_stage_num, name=name+'_pred')  # prediction     |
                                                                                                        #               |
    local_s = fully_act(data=pred1, activation='tanh', Kth_stage_num=Kth_stage_num, name=name+'_local')  # <------------|

    return delta_s, pred_s, local_s

def get_symbol(stage_num, lambda_local=0.25, lambda_d=0.25, has_transform=True):
    ''' 主函数
    # :param data:           mx.Symbol; input is mx.sym.Variable()
    :param stage_num:      list or tuple, SSRNet's Hyperparameter, 3 stage [3,3,3]
    :param lambda_local:   float SSRNet's Hyperparameter
    :param lambda_d:       float SSRNet's Hyperparameter
    :param has_transform:  bool,  default=True
    :return:               create SSRNet
    '''
    data = mx.sym.Variable(name='data')
    # 归一化
    if has_transform:
        data = data - 127.5
        data = data * 0.0078125
    # SSRNet
    # stream 1 32
    x_layer1 = conv_bn_act(data=data, num_filter=32, activation='relu', name='x_layer1', pool_type='avg')
    x_layer2 = conv_bn_act(data=x_layer1, num_filter=32, activation='relu', name='x_layer2', pool_type='avg')
    x_layer3 = conv_bn_act(data=x_layer2, num_filter=32, activation='relu', name='x_layer3', pool_type='avg')
    x = conv_bn_act(data=x_layer3, num_filter=32, activation='relu', name='x', pool_type=None)

    # stream 2 16
    s_layer1 = conv_bn_act(data=data, num_filter=16, activation='tanh', name='s_layer1', pool_type='max')
    s_layer2 = conv_bn_act(data=s_layer1, num_filter=16, activation='tanh', name='s_layer2', pool_type='max')
    s_layer3 = conv_bn_act(data=s_layer2, num_filter=16, activation='tanh', name='s_layer3', pool_type='max')
    s = conv_bn_act(data=s_layer3, num_filter=16, activation='tanh', name='s', pool_type=None)

    # Classifier block
    ''' why not global_avgpool or global_maxpool ??? '''
    # [N,1]   [N,3]    [N,3]
    delta_s1, pred_s1, local_s1 = fusion_block_and_regression(x_layer=x, s_layer=s, Kth_stage_num=stage_num[0], num_filter=10, name='s1', pool_size=None)
    delta_s2, pred_s2, local_s2 = fusion_block_and_regression(x_layer=x_layer2, s_layer=s_layer2, Kth_stage_num=stage_num[1], num_filter=10, name='s2', pool_size=(4,4))
    delta_s3, pred_s3, local_s3 = fusion_block_and_regression(x_layer=x_layer1, s_layer=s_layer1, Kth_stage_num=stage_num[2], num_filter=10, name='s3', pool_size=(8,8))

    # compute age
    i1 = mx.symbol.arange(0, stage_num[0])
    i1 = mx.symbol.expand_dims(i1, axis=0)
    a = mx.symbol.broadcast_add(i1, lambda_local * local_s1) * pred_s1  # 广播操作
    a = mx.symbol.sum(a, axis=1, keepdims=True)
    a = a / (stage_num[0] * (1 + lambda_d * delta_s1))

    i2 = mx.symbol.arange(0, stage_num[1])
    i2 = mx.symbol.expand_dims(i2, axis=0)
    b = mx.symbol.broadcast_add(i2, lambda_local * local_s2) * pred_s2
    b = mx.symbol.sum(b, axis=1, keepdims=True)
    b = b / (stage_num[0] * (1 + lambda_d * delta_s1)) / (stage_num[1] * (1 + lambda_d * delta_s2))

    i3 = mx.symbol.arange(0, stage_num[2])
    i3 = mx.symbol.expand_dims(i3, axis=0)
    c = mx.symbol.broadcast_add(i3, lambda_local * local_s3) * pred_s3
    c = mx.symbol.sum(c, axis=1, keepdims=True)
    c = c / (stage_num[0] * (1 + lambda_d * delta_s1)) / (stage_num[1] * (1 + lambda_d * delta_s2)) / (stage_num[2] * (1 + lambda_d * delta_s3))

    pred_age = 101 * (a + b + c)

    return pred_age


'''
    动态版 Imperative API 命令式
'''
import mxnet as mx
# import mxnet.gluon.model_zoo.vision as model_vision
import mxnet.gluon.data.vision as data_vision
from mxnet.gluon import nn



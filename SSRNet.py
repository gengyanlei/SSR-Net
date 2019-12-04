'''
    implement of SSRNet mxnet version
    (SSR-Net: A Compact Soft Stagewise Regression Network for Age Estimation)[使用了论文部分截图]
    MXNet(静态版Symbolic、动态版Imperative), 推荐 mxnet.gluon
    2D format is [N,C,H,W] !!!   Not [N,H,W,C]
    author is leilei
'''
##############################################################################

__all__ = ['get_symbol_ssrnet', 'get_imperative_ssrnet']

'''
    静态版 Symbolic API 符号式
    其实可以直接使用mx.sym 不必再另创建函数，进行规律寻找组建；这样规律不好寻找；
    为了便于查看，我将截图与函数命名 一致，便于查看组合
'''
import mxnet as mx

def conv_bn_act(data, num_filter, activation, name, pool_type=None):
    '''  对应 Screenshot/conv_bn_act.png
    :param data:        Symbol, input symbol type [N,C,H,W]
    :param num_filter:  int, output_channels
    :param activation:  str, activation functions ['relu', 'sigmoid', 'softrelu', 'softsign', 'tanh']
    :param name:        str, layer's name(Unique)
    :param pool_type:   str, pool's type ['max', 'avg'...], default=None
    :return:            conv2d batch_norm activation pool function
    '''
    conv = mx.sym.Convolution(data=data, kernel=(3,3), stride=(1,1), pad=(1,1), num_filter=num_filter, name=name+'_conv')  # no padding and has bias
    bn = mx.sym.BatchNorm(data=conv, fix_gamma=False, eps=2e-5, momentum=0.9, axis=1, name=name+'_bn')  # pytorch tf keras不需要设置什么参数，默认即可， mxnet则不行
    out = mx.sym.Activation(data=bn, act_type=activation, name=name+'_act')
    if pool_type is not None:
        out = mx.sym.Pooling(out, kernel=(2,2), stride=(2,2), pool_type=pool_type, name=name+'_pool')
    return out

def conv_act_drop_fully(data, num_filter, Kth_stage_num, name, pool_type='avg', pool_size=None):
    '''  对应 Screenshot/conv_act_drop_fully.png
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
        act = mx.sym.Pooling(data=act, kernel=pool_size, stride=pool_size, pool_type=pool_type, name=name+'_pool')
    layer = mx.sym.Flatten(data=act, name=name+'_fla')

    drop = mx.sym.Dropout(data=layer, p=0.2, name=name+'_mix_drop')
    fully = mx.sym.FullyConnected(data=drop, num_hidden=Kth_stage_num, name=name+'_mix_fully')
    layer_mix = mx.sym.Activation(data=fully, act_type='relu', name=name+'_mix_act')

    return layer, layer_mix

def fully_act(data, activation, Kth_stage_num, name):
    '''  对应 Screenshot/fully_act.png
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
    '''  对应 Screenshot/fusion_block_and_regression.png
    :param x_layer:        Symbol stream 1
    :param s_layer:        Symbol stream 2
    :param Kth_stage_num:  int
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

def get_symbol_ssrnet(stage_num, lambda_local=0.25, lambda_d=0.25, has_transform=True):
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
    动态版 Imperative API 命令式  [nn.HybridBlock虽然看懂了API，但是有一些小地方暂时没弄明白，也可以使用]
'''
import mxnet as mx
# import mxnet.gluon.model_zoo.vision as model_vision
import mxnet.gluon.data.vision as data_vision
from mxnet.gluon import nn

def conv_bn_act1(num_filter, activation, pool_type=None):
    '''
    :param num_filter:  int output's channel
    :param activation:  str activation type
    :param pool_type:   str [avg, max]
    :return:
    '''
    net = nn.HybridSequential(nn.Conv2D(channels=num_filter, kernel_size=(3,3), strides=(1,1), padding=(1,1)),
                              nn.BatchNorm(),
                              nn.Activation(activation=activation))
    if pool_type == 'avg':
        net.add( nn.AvgPool2D(pool_size=(2,2), strides=(2,2)) )
    elif pool_type == 'max':
        net.add( nn.MaxPool2D(pool_size=(2,2), strides=(2,2)) )

    return net

def conv_relu_pool_fla(num_filter, pool_type=None, pool_size=None):
    '''
    :param num_filter:  int output channels, 10
    :param pool_type:   str [avg max]
    :param pool_size:   tuple int
    :return:  after flatten
    '''
    net = nn.HybridSequential(nn.Conv2D(channels=num_filter, kernel_size=(1,1), activation='relu'))  # 如果报错， 去掉relu，改用 nn.Activation('relu')
    if pool_type == 'avg' and pool_size is not None:
        net.add( nn.AvgPool2D(pool_size=pool_size, strides=pool_size) )
    elif pool_type == 'max' and pool_size is not None:
        net.add( nn.MaxPool2D(pool_size=pool_size, strides=pool_size) )
    net.add(nn.Flatten())
    return net

def drop_fc_relu(Kth_stage_num):
    '''
    :param Kth_stage_num: int 第K个 stage num
    :return:
    '''
    net = nn.HybridSequential(nn.Dropout(0.2),
                              nn.Dense(units=Kth_stage_num, activation='relu'))  # 若报错 采用nn.Activation('relu')
    return net

def fc_act(Kth_stage_num, activation, has_repeat=False):
    '''
    :param Kth_stage_num:  int 第K个 stage num
    :param activation:     str [relu tanh]
    :param has_repeat:     bool default False
    :return:
    '''
    net = nn.HybridSequential()  # 若报错 采用nn.Activation('relu')
    if has_repeat:
        net.add(nn.Dense(units=2*Kth_stage_num, activation='relu'))  # 这一层 均为relu
    net.add(nn.Dense(units=Kth_stage_num, activation=activation))  # 若报错 采用nn.Activation('relu')
    return net



class SSRNet(nn.HybridBlock):
    def __init__(self, stage_num=[3,3,3], lambda_local=0.25, lambda_d=0.25,):
        '''
        :param stage_num:  default [3,3,3]
        '''
        super().__init__()
        # hyperparameter
        self.stage_num = stage_num
        self.V = 101
        self.lambda_local = lambda_local
        self.lambda_d = lambda_d

        self.x_layer1 = conv_bn_act1(num_filter=32, activation='relu', pool_type='avg')
        self.x_layer2 = conv_bn_act1(num_filter=32, activation='relu', pool_type='avg')
        self.x_layer3 = conv_bn_act1(num_filter=32, activation='relu', pool_type='avg')
        self.x = conv_bn_act1(num_filter=32, activation='relu', pool_type='avg')

        self.s_layer1 = conv_bn_act1(num_filter=16, activation='tanh', pool_type='max')
        self.s_layer1 = conv_bn_act1(num_filter=16, activation='tanh', pool_type='max')
        self.s_layer1 = conv_bn_act1(num_filter=16, activation='tanh', pool_type='max')
        self.s = conv_bn_act1(num_filter=16, activation='tanh', pool_type='max')

        self.x_before_PB = conv_relu_pool_fla(num_filter=10, pool_type=None, pool_size=None)  # has flatten
        self.x2_before_PB = conv_relu_pool_fla(num_filter=10, pool_type='avg', pool_size=(4,4))
        self.x1_before_PB = conv_relu_pool_fla(num_filter=10, pool_type='avg', pool_size=(8,8))  # has flatten

        self.s_before_PB = conv_relu_pool_fla(num_filter=10, pool_type=None, pool_size=None)  # has flatten
        self.s2_before_PB = conv_relu_pool_fla(num_filter=10, pool_type='max', pool_size=(4,4))
        self.s1_before_PB = conv_relu_pool_fla(num_filter=10, pool_type='max', pool_size=(8,8))  # has flatten

        self.x_PB = drop_fc_relu(Kth_stage_num=self.stage_num[0])  # no need flatten
        self.x2_PB = drop_fc_relu(Kth_stage_num=self.stage_num[1])
        self.x1_PB = drop_fc_relu(Kth_stage_num=self.stage_num[2])  # no need flatten

        self.s_PB = drop_fc_relu(Kth_stage_num=self.stage_num[0])  # no need flatten
        self.s2_PB = drop_fc_relu(Kth_stage_num=self.stage_num[1])
        self.s1_PB = drop_fc_relu(Kth_stage_num=self.stage_num[2])  # no need flatten
        # delta_s1, pred_s1, local_s1
        self.delta_s1 = fc_act(Kth_stage_num=1, activation='tanh', has_repeat=False)
        self.pred_s1 = fc_act(Kth_stage_num=self.stage_num[0], activation='relu', has_repeat=True)
        self.local_s1 = fc_act(Kth_stage_num=self.stage_num[0], activation='tanh', has_repeat=True)
        # delta_s2, pred_s2, local_s2
        self.delta_s2 = fc_act(Kth_stage_num=1, activation='tanh', has_repeat=False)
        self.pred_s2 = fc_act(Kth_stage_num=self.stage_num[1], activation='relu', has_repeat=True)
        self.local_s2 = fc_act(Kth_stage_num=self.stage_num[1], activation='tanh', has_repeat=True)
        # delta_s3, pred_s3, local_s3
        self.delta_s3 = fc_act(Kth_stage_num=1, activation='tanh', has_repeat=False)
        self.pred_s3 = fc_act(Kth_stage_num=self.stage_num[2], activation='relu', has_repeat=True)
        self.local_s3 = fc_act(Kth_stage_num=self.stage_num[2], activation='tanh', has_repeat=True)

    def hybrid_forward(self, F, x):
        x_layer1 = self.x_layer1(x)
        x_layer2 = self.x_layer2(x_layer1)
        x_layer3 = self.x_layer3(x_layer2)
        x_ = self.x(x_layer3)

        s_layer1 = self.s_layer1(x)
        s_layer2 = self.s_layer2(s_layer1)
        s_layer3 = self.s_layer3(s_layer2)
        s_ = self.s(s_layer3)

        x_before_PB = self.x_before_PB(x_)          # has flatten
        x2_before_PB = self.x2_before_PB(x_layer2)
        x1_before_PB = self.x1_before_PB(x_layer1)

        s_before_PB = self.s_before_PB(s_)          # has flatten
        s2_before_PB = self.s2_before_PB(s_layer2)
        s1_before_PB = self.s1_before_PB(s_layer1)

        x_PB = self.x_PB(x_before_PB)
        x2_PB = self.x2_PB(x2_before_PB)
        x1_PB = self.x1_PB(x1_before_PB)

        s_PB = self.s_PB(s_before_PB)
        s2_PB = self.s2_PB(s2_before_PB)
        s1_PB = self.s1_PB(s1_before_PB)

        delta_s1= self.delta_s1(x_before_PB*s_before_PB)  # 若报错， 修改成 F.broadcast_mul(x_before_PB, s_before_PB)
        pred_s1= self.pred_s1(x_PB*s_PB)
        local_s1= self.local_s1(x_PB*s_PB)

        delta_s2 = self.delta_s2(x2_before_PB*s2_before_PB)  # 若报错， 修改成 F.broadcast_mul(, )
        pred_s2 = self.pred_s2(x2_PB*s2_PB)
        local_s2 = self.local_s2(x2_PB*s2_PB)

        delta_s3 = self.delta_s3(x1_before_PB*s1_before_PB)  # 若报错， 修改成 F.broadcast_mul(, )
        pred_s3 = self.pred_s3(x1_PB*s1_PB)
        local_s3 = self.local_s3(x1_PB*s1_PB)

        # 采用 F. 操作
        # compute age
        i1 = F.arange(0, self.stage_num[0])
        i1 = F.expand_dims(i1, axis=0)
        a = F.broadcast_add(i1, self.lambda_local * local_s1) * pred_s1  # 广播操作
        a = F.sum(a, axis=1, keepdims=True)
        a = a / (self.stage_num[0] * (1 + self.lambda_d * delta_s1))

        i2 = F.arange(0, self.stage_num[1])
        i2 = F.expand_dims(i2, axis=0)
        b = F.broadcast_add(i2, self.lambda_local * local_s2) * pred_s2
        b = F.sum(b, axis=1, keepdims=True)
        b = b / (self.stage_num[0] * (1 + self.lambda_d * delta_s1)) / (self.stage_num[1] * (1 + self.lambda_d * delta_s2))

        i3 = F.arange(0, self.stage_num[2])
        i3 = F.expand_dims(i3, axis=0)
        c = F.broadcast_add(i3, self.lambda_local * local_s3) * pred_s3
        c = F.sum(c, axis=1, keepdims=True)
        c = c / (self.stage_num[0] * (1 + self.lambda_d * delta_s1)) / (self.stage_num[1] * (1 + self.lambda_d * delta_s2)) / (self.stage_num[2] * (1 + self.lambda_d * delta_s3))

        pred_age = 101 * (a + b + c)

        return pred_age


def get_imperative_ssrnet(stage_num, lambda_local=0.25, lambda_d=0.25):
    '''
    :param stage_num:     list or tuple, SSRNet's Hyperparameter, 3 stage [3,3,3]
    :param lambda_local:  float SSR-Net hyperparameter
    :param lambda_d:      float SSR-Net hyperparameter
    :return:
    '''
    net = SSRNet(stage_num=stage_num, lambda_local=lambda_local, lambda_d=lambda_d)
    return net






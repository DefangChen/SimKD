from .resnet import resnet38, resnet110, resnet116, resnet14x2, resnet38x2, resnet110x2
from .resnet import resnet8x4, resnet14x4, resnet32x4, resnet38x4
from .vgg import vgg8_bn, vgg13_bn
from .mobilenetv2 import mobile_half, mobile_half_double
from .ShuffleNetv1 import ShuffleV1
from .ShuffleNetv2 import ShuffleV2, ShuffleV2_1_5

from .resnet_imagenet import resnet18, resnet34, resnet50, wide_resnet50_2, resnext50_32x4d
from .resnet_imagenet import wide_resnet10_2, wide_resnet18_2, wide_resnet34_2
from .mobilenetv2_imagenet import mobilenet_v2
from .shuffleNetv2_imagenet import shufflenet_v2_x1_0

model_dict = {
    'resnet38': resnet38,
    'resnet110': resnet110,
    'resnet116': resnet116,
    'resnet14x2': resnet14x2,
    'resnet38x2': resnet38x2,
    'resnet110x2': resnet110x2,
    'resnet8x4': resnet8x4,
    'resnet14x4': resnet14x4,
    'resnet32x4': resnet32x4,
    'resnet38x4': resnet38x4,
    'vgg8': vgg8_bn,
    'vgg13': vgg13_bn,
    'MobileNetV2': mobile_half,
    'MobileNetV2_1_0': mobile_half_double,
    'ShuffleV1': ShuffleV1,
    'ShuffleV2': ShuffleV2,
    'ShuffleV2_1_5': ShuffleV2_1_5,
    
    'ResNet18': resnet18,
    'ResNet34': resnet34,
    'ResNet50': resnet50,
    'resnext50_32x4d': resnext50_32x4d,
    'ResNet10x2': wide_resnet10_2,
    'ResNet18x2': wide_resnet18_2,
    'ResNet34x2': wide_resnet34_2,
    'wrn_50_2': wide_resnet50_2,
    
    'MobileNetV2_Imagenet': mobilenet_v2,
    'ShuffleV2_Imagenet': shufflenet_v2_x1_0,
}

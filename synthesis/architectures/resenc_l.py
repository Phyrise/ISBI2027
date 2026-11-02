"""ResEnc-L: the nnFoundationCNN backbone (nnssl ``get_res_enc_l``), used for the AFP teachers and the two segmentation
experts. ``dynamic_network_architectures`` ResidualEncoderUNet, 6 stages 32-64-128-256-320-320, strides 1-2-2-2-2-2,
blocks 1-3-4-6-6-6, one decoder conv per stage, InstanceNorm (affine), LeakyReLU, conv bias, deep supervision off.
num_classes = 1 for the nnssl MAE checkpoint, 2 for the lesion experts.
"""
from torch import nn

RESENC_L = dict(n_stages=6, features_per_stage=[32, 64, 128, 256, 320, 320], kernel_sizes=[[3, 3, 3]] * 6,
                strides=[[1, 1, 1]] + [[2, 2, 2]] * 5, n_blocks_per_stage=[1, 3, 4, 6, 6, 6],
                n_conv_per_stage_decoder=[1, 1, 1, 1, 1], conv_bias=True)


def build_resenc_l(input_channels=1, num_classes=1):
    from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet
    return ResidualEncoderUNet(input_channels=input_channels, conv_op=nn.Conv3d, num_classes=num_classes,
                               norm_op=nn.InstanceNorm3d, norm_op_kwargs={'eps': 1e-5, 'affine': True},
                               nonlin=nn.LeakyReLU, nonlin_kwargs={'inplace': True}, deep_supervision=False,
                               **RESENC_L)

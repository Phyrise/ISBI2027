"""SwinUNETR generator (MONAI 1.5.1 ``monai.networks.nets.SwinUNETR``): 3D, 1 input / 1 output channel,
feature_size 24, gradient checkpointing on (memory only; it does not change the maths)."""


def build_swinunetr(feature_size=24, use_checkpoint=True):
    from monai.networks.nets import SwinUNETR
    return SwinUNETR(in_channels=1, out_channels=1, feature_size=int(feature_size), spatial_dims=3,
                     use_checkpoint=bool(use_checkpoint))

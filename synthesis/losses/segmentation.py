"""Lesion-expert loss: the nnU-Net Dice + cross-entropy (``nnunetv2.training.loss.compound_losses.DC_and_CE_loss``)
with batch Dice, background excluded from the Dice term, smooth 1e-5, ``MemoryEfficientSoftDiceLoss``, and label -1
(outside the valid volume) ignored. Only the loss comes from nnU-Net; the experts are trained by this repository's own
loop (synthesis/trainers/sgd_trainer.py), not by a stock nnUNetv2 trainer."""


def segmentation_loss():
    from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
    from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
    return DC_and_CE_loss({'batch_dice': True, 'do_bg': False, 'smooth': 1e-5, 'ddp': False}, {},
                          ignore_label=-1, dice_class=MemoryEfficientSoftDiceLoss)

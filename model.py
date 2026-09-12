"""EfficientNet-B3 DR grader (5-class softmax)."""
import timm

# Unfrozen when freeze=True: last two MBConv stages + head. Everything earlier keeps ImageNet features.
_TRAINABLE = ("blocks.5", "blocks.6", "conv_head", "bn2", "classifier")


def build_model(num_classes=5, pretrained=True, freeze=True):
    model = timm.create_model("efficientnet_b3", pretrained=pretrained, num_classes=num_classes)
    if freeze:
        for name, param in model.named_parameters():
            if not any(k in name for k in _TRAINABLE):
                param.requires_grad = False
    return model


def target_layer(model):
    """blocks[4]: 19x19 at 300x300 input, the layer Grad-CAM hooks.

    Not the deepest option (conv_head, 10x10, see target_layer_coarse) but a controlled
    comparison across grades 0/1/2/3/4 showed conv_head's heatmaps were dominated by a few
    giant blobs that often hugged the image border; blocks[4]'s ~3.6x finer grid resolves into
    many small, discrete hotspots that land on the optic disc, vessel arcades, and scattered
    lesion-like points instead - visibly more clinically plausible at the same compute cost.
    See README Explainability section.
    """
    return model.blocks[4]


def target_layer_coarse(model):
    """conv_head: 10x10 at 300x300 input - the deepest, most class-semantic features, but a
    coarser Grad-CAM grid. Kept for comparison; target_layer() is the default used in practice."""
    return model.conv_head

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
    """Last conv block feeding the classifier — the layer Grad-CAM hooks."""
    return model.conv_head

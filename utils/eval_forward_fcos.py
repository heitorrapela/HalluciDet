from typing import Tuple, List, Dict, Optional
import torch
from torch import Tensor
from collections import OrderedDict
import torch.nn.functional as F

# Modified by us
# Modified code from torchvision to calculate loss during evaluation for FCOS, with this we can weight
# Original code: https://github.com/pytorch/vision/blob/main/torchvision/models/detection/fcos.py

def eval_forward_fcos(model, images, targets, train_det=False, model_name='ssd', debug=False):

    if not train_det:
        model.eval()

    for target in targets:
        boxes = target["boxes"]
        if isinstance(boxes, torch.Tensor):
            torch._assert(
                len(boxes.shape) == 2 and boxes.shape[-1] == 4,
                f"Expected target boxes to be a tensor of shape [N, 4], got {boxes.shape}.",
            )
        else:
            torch._assert(False, f"Expected target boxes to be of type Tensor, got {type(boxes)}.")

    original_image_sizes: List[Tuple[int, int]] = []
    for img in images:
        val = img.shape[-2:]
        torch._assert(
            len(val) == 2,
            f"expecting the last two dimensions of the Tensor to be H and W instead got {img.shape[-2:]}",
        )
        original_image_sizes.append((val[0], val[1]))

    # transform the input
    images, targets = model.transform(images, targets)

    # Check for degenerate boxes
    if targets is not None:
        for target_idx, target in enumerate(targets):
            boxes = target["boxes"]
            degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
            if degenerate_boxes.any():
                # print the first degenerate box
                bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                degen_bb: List[float] = boxes[bb_idx].tolist()
                torch._assert(
                    False,
                    "All bounding boxes should have positive height and width."
                    f" Found invalid box {degen_bb} for target at index {target_idx}.",
                )

    features = model.backbone(images.tensors)
    if isinstance(features, torch.Tensor):
        features = OrderedDict([("0", features)])

    features = list(features.values())

    # compute the fcos heads outputs using the features
    head_outputs = model.head(features)

    # create the set of anchors
    anchors = model.anchor_generator(images, features)
    # recover level sizes
    num_anchors_per_level = [x.size(2) * x.size(3) for x in features]

    losses = {}
    detections: List[Dict[str, Tensor]] = []

    # compute the losses
    losses = model.compute_loss(targets, head_outputs, anchors, num_anchors_per_level)

    # split outputs per level
    split_head_outputs: Dict[str, List[Tensor]] = {}
    for k in head_outputs:
        split_head_outputs[k] = list(head_outputs[k].split(num_anchors_per_level, dim=1))
    split_anchors = [list(a.split(num_anchors_per_level)) for a in anchors]

    # compute the detections
    detections = model.postprocess_detections(split_head_outputs, split_anchors, images.image_sizes)
    detections = model.transform.postprocess(detections, images.image_sizes, original_image_sizes)

    return losses, detections
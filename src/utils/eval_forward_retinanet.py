import math
import warnings
from collections import OrderedDict
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple
import torchvision
import torch


# Modified by us
# Modified code from torchvision to calculate loss during evaluation for Retinanet, with this we can train our model with the frozen detector loss
# Original code: https://github.com/pytorch/vision/blob/main/torchvision/models/detection/retinanet.py
# https://github.com/pytorch/vision/blob/14553fb9eb78e053100e966feee6149905f9922c/torchvision/ops/focal_loss.py#L7

# Helper functions for Retinanet
def _sum(x: List[torch.Tensor]) -> torch.Tensor:
    res = x[0]
    for i in x[1:]:
        res = res + i
    return res

def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2,
    reduction: str = "none",
) -> torch.Tensor:
   
    p = torch.sigmoid(inputs)
    ce_loss = torch.nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    # Check reduction option and return loss accordingly
    if reduction == "none":
        pass
    elif reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    else:
        raise ValueError(
            f"Invalid Value for arg 'reduction': '{reduction} \n Supported reduction modes: 'none', 'mean', 'sum'"
        )
    return loss


def box_loss(
    type: str,
    box_coder: torchvision.models.detection._utils.BoxCoder,
    anchors_per_image: torch.Tensor,
    matched_gt_boxes_per_image: torch.Tensor,
    bbox_regression_per_image: torch.Tensor,
    cnf: Optional[Dict[str, float]] = None,
) -> torch.Tensor:
    torch._assert(type in ["l1", "smooth_l1", "ciou", "diou", "giou"], f"Unsupported loss: {type}")

    if type == "l1":
        target_regression = box_coder.encode_single(matched_gt_boxes_per_image, anchors_per_image)
        return torch.nn.functional.l1_loss(bbox_regression_per_image, target_regression, reduction="sum")
    elif type == "smooth_l1":
        
        target_regression = box_coder.encode_single(matched_gt_boxes_per_image, anchors_per_image)
        beta = cnf["beta"] if cnf is not None and "beta" in cnf else 1.0
        return torch.nn.functional.smooth_l1_loss(bbox_regression_per_image, target_regression, reduction="sum", beta=beta)
    else:

        bbox_per_image = box_coder.decode_single(bbox_regression_per_image, anchors_per_image)
        eps = cnf["eps"] if cnf is not None and "eps" in cnf else 1e-7
        if type == "ciou":
            return torchvision.ops.complete_box_iou_loss(bbox_per_image, matched_gt_boxes_per_image, reduction="sum", eps=eps)
        if type == "diou":
            return torchvision.ops.distance_box_iou_loss(bbox_per_image, matched_gt_boxes_per_image, reduction="sum", eps=eps)
        # otherwise giou
        return torchvision.ops.generalized_box_iou_loss(bbox_per_image, matched_gt_boxes_per_image, reduction="sum", eps=eps)


def eval_forward_retinanet(model, images, targets, train_det=False, model_name='retinanet'):

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

    head_outputs = model.head(features)

    anchors = model.anchor_generator(images, features)

    losses = {}
    detections: List[Dict[str, torch.Tensor]] = []

    losses = compute_retinanet_loss(targets, head_outputs, anchors, model)
    # compute_retinanet_loss(targets, head_outputs, anchors, matched_idxs, model)

    # recover level sizes
    num_anchors_per_level = [x.size(2) * x.size(3) for x in features]
    HW = 0
    for v in num_anchors_per_level:
        HW += v
    HWA = head_outputs["cls_logits"].size(1)
    A = HWA // HW
    num_anchors_per_level = [hw * A for hw in num_anchors_per_level]

    # split outputs per level
    split_head_outputs: Dict[str, List[torch.Tensor]] = {}
    for k in head_outputs:
        split_head_outputs[k] = list(head_outputs[k].split(num_anchors_per_level, dim=1))
    split_anchors = [list(a.split(num_anchors_per_level)) for a in anchors]

    # compute the detections
    detections = model.postprocess_detections(split_head_outputs, split_anchors, images.image_sizes)
    detections = model.transform.postprocess(detections, images.image_sizes, original_image_sizes)

    return losses, detections
    

def compute_retinanet_loss(targets, head_outputs, anchors, model):
    matched_idxs = []

    for anchors_per_image, targets_per_image in zip(anchors, targets):
        if targets_per_image["boxes"].numel() == 0:
            matched_idxs.append(
                torch.full((anchors_per_image.size(0),), -1, dtype=torch.int64, device=anchors_per_image.device)
            )
            continue

        match_quality_matrix = torchvision.ops.box_iou(targets_per_image["boxes"], anchors_per_image)
        matched_idxs.append(model.proposal_matcher(match_quality_matrix))

    return {
        "classification": compute_loss_classification_head(targets, head_outputs, matched_idxs, model),
        "bbox_regression": compute_loss_regression_head(targets, head_outputs, anchors, matched_idxs, model),
    }

def compute_loss_classification_head(targets, head_outputs, matched_idxs, model):
    losses = []

    cls_logits = head_outputs["cls_logits"]

    for targets_per_image, cls_logits_per_image, matched_idxs_per_image in zip(targets, cls_logits, matched_idxs):
        # determine only the foreground
        foreground_idxs_per_image = matched_idxs_per_image >= 0
        num_foreground = foreground_idxs_per_image.sum()

        # create the target classification
        gt_classes_target = torch.zeros_like(cls_logits_per_image)
        gt_classes_target[
            foreground_idxs_per_image,
            targets_per_image["labels"][matched_idxs_per_image[foreground_idxs_per_image]],
        ] = 1.0

        # find indices for which anchors should be ignored
        valid_idxs_per_image = matched_idxs_per_image != model.head.classification_head.BETWEEN_THRESHOLDS

        # compute the classification loss
        losses.append(
            sigmoid_focal_loss(
                cls_logits_per_image[valid_idxs_per_image],
                gt_classes_target[valid_idxs_per_image],
                reduction="sum",
            )
            / max(1, num_foreground)
        )

    return _sum(losses) / len(targets)



def compute_loss_regression_head(targets, head_outputs, anchors, matched_idxs, model, loss_reg='smooth_l1'):
    losses = []

    bbox_regression = head_outputs["bbox_regression"]

    for targets_per_image, bbox_regression_per_image, anchors_per_image, matched_idxs_per_image in zip(
        targets, bbox_regression, anchors, matched_idxs
    ):
        # determine only the foreground indices, ignore the rest
        foreground_idxs_per_image = torch.where(matched_idxs_per_image >= 0)[0]
        num_foreground = foreground_idxs_per_image.numel()

        # select only the foreground boxes
        matched_gt_boxes_per_image = targets_per_image["boxes"][matched_idxs_per_image[foreground_idxs_per_image]]
        bbox_regression_per_image = bbox_regression_per_image[foreground_idxs_per_image, :]
        anchors_per_image = anchors_per_image[foreground_idxs_per_image, :]

        # compute the loss
        losses.append(
            box_loss(
                loss_reg,
                model.box_coder,
                anchors_per_image,
                matched_gt_boxes_per_image,
                bbox_regression_per_image,
            )
            / max(1, num_foreground)
        )

    return _sum(losses) / max(1, len(targets))
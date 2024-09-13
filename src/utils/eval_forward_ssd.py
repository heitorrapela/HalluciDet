from typing import Tuple, List, Dict, Optional
import torch
from torch import Tensor
from collections import OrderedDict
import torch.nn.functional as F
import numpy as np
import cv2
from src.config.config import Config


# implementation from https://github.com/kuangliu/torchcv/blob/master/torchcv/utils/box.py
def _upcast(t: Tensor) -> Tensor:
    # Protects from numerical overflows in multiplications by upcasting to the equivalent higher type
    if t.is_floating_point():
        return t if t.dtype in (torch.float32, torch.float64) else t.float()
    else:
        return t if t.dtype in (torch.int32, torch.int64) else t.int()


def box_area(boxes: Tensor) -> Tensor:
    """
    Computes the area of a set of bounding boxes, which are specified by their
    (x1, y1, x2, y2) coordinates.
    Args:
        boxes (Tensor[N, 4]): boxes for which the area will be computed. They
            are expected to be in (x1, y1, x2, y2) format with
            ``0 <= x1 < x2`` and ``0 <= y1 < y2``.
    Returns:
        Tensor[N]: the area for each box
    """
    # if not torch.jit.is_scripting() and not torch.jit.is_tracing():
    #     _log_api_usage_once(box_area)
    boxes = _upcast(boxes)
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def _box_inter_union(boxes1: Tensor, boxes2: Tensor) -> Tuple[Tensor, Tensor]:
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = _upcast(rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    return inter, union


def box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """
    Return intersection-over-union (Jaccard index) between two sets of boxes.
    Both sets of boxes are expected to be in ``(x1, y1, x2, y2)`` format with
    ``0 <= x1 < x2`` and ``0 <= y1 < y2``.
    Args:
        boxes1 (Tensor[N, 4]): first set of boxes
        boxes2 (Tensor[M, 4]): second set of boxes
    Returns:
        Tensor[N, M]: the NxM matrix containing the pairwise IoU values for every element in boxes1 and boxes2
    """
    # if not torch.jit.is_scripting() and not torch.jit.is_tracing():
    #     _log_api_usage_once(box_iou)
    inter, union = _box_inter_union(boxes1, boxes2)
    iou = inter / union
    return iou


def eval_forward_ssd(model, images, targets, train_det=False, model_name='ssd', debug=False):
    
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

    # get the original image sizes
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
                bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                degen_bb: List[float] = boxes[bb_idx].tolist()
                torch._assert(
                    False,
                    "All bounding boxes should have positive height and width."
                    f" Found invalid box {degen_bb} for target at index {target_idx}.",
                )

    # get the features from the backbone
    features = model.backbone(images.tensors)
    if isinstance(features, torch.Tensor):
        features = OrderedDict([("0", features)])

    features = list(features.values())

    # compute the ssd heads outputs using the features
    head_outputs = model.head(features)

    # create the set of anchors
    anchors = model.anchor_generator(images, features)

    losses = {}
    detections: List[Dict[str, Tensor]] = []
    matched_idxs = []

    for anchors_per_image, targets_per_image in zip(anchors, targets):
        if targets_per_image["boxes"].numel() == 0:
            matched_idxs.append(
                torch.full(
                    (anchors_per_image.size(0),), -1, dtype=torch.int64, device=anchors_per_image.device
                )
            )
            continue

        match_quality_matrix = box_iou(targets_per_image["boxes"], anchors_per_image)
        matched_idxs.append(model.proposal_matcher(match_quality_matrix))

    # losses_original = model.compute_loss(targets, head_outputs, anchors, matched_idxs)
    losses = compute_loss(model, targets, head_outputs, anchors, matched_idxs, train_det=train_det)  

    detections = model.postprocess_detections(head_outputs, anchors, images.image_sizes)
    detections = model.transform.postprocess(detections, images.image_sizes, original_image_sizes)
    
    losses['masked'] = torch.tensor(0.0)

    if(Config.Losses.hparams_losses_weights['det_masked'] > 0.0):
        images_cpy = images.copy()
        targets_cpy = targets.copy()
        
        loss_debug = torch.nn.MSELoss()
        total_fg = 0
        for idx in range(len(targets_cpy)):
            img = bbox2image(targets_cpy[idx]['boxes'], size=(images_cpy[idx].shape[1], images_cpy[idx].shape[2]))
            masked_fg = (torch.Tensor(img).unsqueeze(0).repeat(3,1,1)) * images_cpy[idx].cpu()
            total_fg += loss_debug(images_cpy[idx], masked_fg.to(images_cpy[idx].device))
        losses['masked'] = total_fg
        

    return losses, detections

def compute_loss(
    model,
    targets: List[Dict[str, Tensor]],
    head_outputs: Dict[str, Tensor],
    anchors: List[Tensor],
    matched_idxs: List[Tensor],
    train_det=False,
) -> Dict[str, Tensor]:
    bbox_regression = head_outputs["bbox_regression"]
    cls_logits = head_outputs["cls_logits"]

    # Match original targets with default boxes
    num_foreground = 0
    bbox_loss = []
    cls_targets = []
    for (
        targets_per_image,
        bbox_regression_per_image,
        cls_logits_per_image,
        anchors_per_image,
        matched_idxs_per_image,
    ) in zip(targets, bbox_regression, cls_logits, anchors, matched_idxs):
        # produce the matching between boxes and targets
        foreground_idxs_per_image = torch.where(matched_idxs_per_image >= 0)[0]
        foreground_matched_idxs_per_image = matched_idxs_per_image[foreground_idxs_per_image]
        num_foreground += foreground_matched_idxs_per_image.numel()

        # Calculate regression loss
        matched_gt_boxes_per_image = targets_per_image["boxes"][foreground_matched_idxs_per_image]
        bbox_regression_per_image = bbox_regression_per_image[foreground_idxs_per_image, :]
        anchors_per_image = anchors_per_image[foreground_idxs_per_image, :]
        target_regression = model.box_coder.encode_single(matched_gt_boxes_per_image, anchors_per_image)

        bbox_loss.append(
            torch.nn.functional.smooth_l1_loss(bbox_regression_per_image, target_regression, reduction="sum")
        )

        # Estimate ground truth for class targets
        gt_classes_target = torch.zeros(
            (cls_logits_per_image.size(0),),
            dtype=targets_per_image["labels"].dtype,
            device=targets_per_image["labels"].device,
        )
        gt_classes_target[foreground_idxs_per_image] = targets_per_image["labels"][
            foreground_matched_idxs_per_image
        ]
        cls_targets.append(gt_classes_target)

    bbox_loss = torch.stack(bbox_loss)
    cls_targets = torch.stack(cls_targets)

    # Calculate classification loss
    num_classes = cls_logits.size(-1)

    # Dont apply label smothing
    if(train_det):
        cls_loss = F.cross_entropy(cls_logits.view(-1, num_classes), cls_targets.view(-1), reduction="none",
                                label_smoothing=Config.Losses.label_smoothing
                                ).view(
            cls_targets.size()
        )
    else: # Dont apply label smothing
        cls_loss = F.cross_entropy(cls_logits.view(-1, num_classes), cls_targets.view(-1), reduction="none",
                                ).view(
            cls_targets.size()
        )

    # Hard Negative Sampling
    foreground_idxs = cls_targets > 0
    num_negative = model.neg_to_pos_ratio * foreground_idxs.sum(1, keepdim=True)
    # num_negative[num_negative < self.neg_to_pos_ratio] = self.neg_to_pos_ratio
    negative_loss = cls_loss.clone()
    negative_loss[foreground_idxs] = -float("inf")  # use -inf to detect positive values that creeped in the sample
    values, idx = negative_loss.sort(1, descending=True)
    # background_idxs = torch.logical_and(idx.sort(1)[1] < num_negative, torch.isfinite(values))
    background_idxs = idx.sort(1)[1] < num_negative

    N = max(1, num_foreground)
    return {
        "bbox_regression": bbox_loss.sum() / N,
        "classification": (cls_loss[foreground_idxs].sum() + cls_loss[background_idxs].sum()) / N,
    }


def bbox2ndimage(bbox, size, class_id=1):
    number_boxes = bbox.shape[0]
    ndimg = np.zeros((number_boxes, size[0], size[1]))
    for i in range(number_boxes):
        x1, y1, x2, y2 = tuple(bbox[i, :].tolist())
        ndimg[i, int(y1) : int(y2) + 1, int(x1) : int(x2) + 1] = class_id

    return ndimg


def bbox2image(bbox, size, class_id=1):
    number_boxes = bbox.shape[0]
    img = np.zeros((size[0], size[1]))
    for i in range(number_boxes):
        x1, y1, x2, y2 = tuple(bbox[i, :].tolist())
        img[int(y1) : int(y2) + 1, int(x1) : int(x2) + 1] = class_id
    return img

def ndimage2bbox(ndimg):
    number_boxes = ndimg.shape[2]
    bbox = np.zeros((number_boxes, 5))
    for i in range(number_boxes):
        chn = ndimg[:, :, i]
        class_id = chn.max()
        if class_id > 0:
            index = np.where(chn != 0)
            x1, x2 = np.min(index[1]), np.max(index[1])
            y1, y2 = np.min(index[0]), np.max(index[0])
            bbox[i, :] = np.array([x1, y1, x2, y2, class_id - 1])
        else:
            bbox[i, :] = np.array([-1, -1, -1, -1, -1])

    valid_boxes = np.where(bbox[:, 4] != -1)[0]
    bbox = bbox[valid_boxes, :] if valid_boxes.size > 0 else np.array([])

    return bbox




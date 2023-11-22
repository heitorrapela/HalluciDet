from typing import Tuple, List, Dict, Optional
import torch
from torch import Tensor
from collections import OrderedDict
import torch.nn.functional as F
from torchvision.models.detection.roi_heads import fastrcnn_loss
from torchvision.models.detection.rpn import concat_box_prediction_layers

# Modified by us
# Modified code from torchvision to calculate loss during evaluation for FasterRCNN, with this we can weight
# Original code: https://github.com/pytorch/vision/tree/main/torchvision/models/detection

def eval_forward_fasterrcnn(model, images, targets, train_det=False, model_name='ssd', debug=False):

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


    proposals, proposal_losses = rpn_eval(model, images, features, targets) 
    detections, detector_losses = roi_heads_eval(model, features, proposals, images.image_sizes, targets)
    detections = model.transform.postprocess(detections, images.image_sizes, original_image_sizes)  # type: ignore[operator]

    losses = {}
    losses.update(detector_losses)
    losses.update(proposal_losses)

    return losses, detections



def rpn_eval(model, images, features, targets):

    # RPN uses all feature maps that are available
    features = list(features.values())
    objectness, pred_bbox_deltas = model.rpn.head(features)
    anchors = model.rpn.anchor_generator(images, features)

    num_images = len(anchors)
    num_anchors_per_level_shape_tensors = [o[0].shape for o in objectness]
    num_anchors_per_level = [s[0] * s[1] * s[2] for s in num_anchors_per_level_shape_tensors]
    objectness, pred_bbox_deltas = concat_box_prediction_layers(objectness, pred_bbox_deltas)
    # apply pred_bbox_deltas to anchors to obtain the decoded proposals
    # note that we detach the deltas because Faster R-CNN do not backprop through
    # the proposals
    proposals = model.rpn.box_coder.decode(pred_bbox_deltas.detach(), anchors)
    proposals = proposals.view(num_images, -1, 4)
    boxes, scores = model.rpn.filter_proposals(proposals, objectness, images.image_sizes, num_anchors_per_level)

    losses = {}
    if targets is None:
        raise ValueError("targets should not be None")
    labels, matched_gt_boxes = model.rpn.assign_targets_to_anchors(anchors, targets)
    regression_targets = model.rpn.box_coder.encode(matched_gt_boxes, anchors)
    loss_objectness, loss_rpn_box_reg = model.rpn.compute_loss(
        objectness, pred_bbox_deltas, labels, regression_targets
    )
    losses = {
        "loss_objectness": loss_objectness,
        "loss_rpn_box_reg": loss_rpn_box_reg,
    }
    return boxes, losses


def roi_heads_eval(model, features, proposals, image_shapes, targets=None, train_det=False):

    if targets is not None:
        for t in targets:
            # TODO: https://github.com/pytorch/pytorch/issues/26731
            floating_point_types = (torch.float, torch.double, torch.half)
            if not t["boxes"].dtype in floating_point_types:
                raise TypeError(f"target boxes must of float type, instead got {t['boxes'].dtype}")
            if not t["labels"].dtype == torch.int64:
                raise TypeError(f"target labels must of int64 type, instead got {t['labels'].dtype}")
            if model.roi_heads.has_keypoint():
                if not t["keypoints"].dtype == torch.float32:
                    raise TypeError(f"target keypoints must of float type, instead got {t['keypoints'].dtype}")

    
    proposals, matched_idxs, labels, regression_targets = model.roi_heads.select_training_samples(proposals, targets)
    
    box_features = model.roi_heads.box_roi_pool(features, proposals, image_shapes)
    box_features = model.roi_heads.box_head(box_features)
    class_logits, box_regression = model.roi_heads.box_predictor(box_features)

    result: List[Dict[str, torch.Tensor]] = []

    losses = {}
    if labels is None:
        raise ValueError("labels cannot be None")
    if regression_targets is None:
        raise ValueError("regression_targets cannot be None")
    loss_classifier, loss_box_reg = fastrcnn_loss(class_logits, box_regression, labels, regression_targets)
    losses = {"loss_classifier": loss_classifier, "loss_box_reg": loss_box_reg}
    
    boxes, scores, labels = model.roi_heads.postprocess_detections(class_logits, box_regression, proposals, image_shapes)
    num_images = len(boxes)
    for i in range(num_images):
        result.append(
            {
                "boxes": boxes[i],
                "labels": labels[i],
                "scores": scores[i],
            }
        )


    if (
        model.roi_heads.keypoint_roi_pool is not None
        and model.roi_heads.keypoint_head is not None
        and model.roi_heads.keypoint_predictor is not None
    ):
        keypoint_proposals = [p["boxes"] for p in result]
        

        # during training, only focus on positive boxes
        num_images = len(proposals)
        keypoint_proposals = []
        pos_matched_idxs = []
        if matched_idxs is None:
            raise ValueError("if in trainning, matched_idxs should not be None")

        for img_id in range(num_images):
            pos = torch.where(labels[img_id] > 0)[0]
            keypoint_proposals.append(proposals[img_id][pos])
            pos_matched_idxs.append(matched_idxs[img_id][pos])

        keypoint_features = model.roi_heads.keypoint_roi_pool(features, keypoint_proposals, image_shapes)
        keypoint_features = model.roi_heads.keypoint_head(keypoint_features)
        keypoint_logits = model.roi_heads.keypoint_predictor(keypoint_features)

        loss_keypoint = {}
        
        if targets is None or pos_matched_idxs is None:
            raise ValueError("both targets and pos_matched_idxs should not be None when in training mode")

        gt_keypoints = [t["keypoints"] for t in targets]
        rcnn_loss_keypoint = keypointrcnn_loss(
            keypoint_logits, keypoint_proposals, gt_keypoints, pos_matched_idxs, train_det
        )
        loss_keypoint = {"loss_keypoint": rcnn_loss_keypoint}

        losses.update(loss_keypoint)

    return result, losses


def keypointrcnn_loss(keypoint_logits, proposals, gt_keypoints, keypoint_matched_idxs, train_det):
    # type: (Tensor, List[Tensor], List[Tensor], List[Tensor]) -> Tensor
    N, K, H, W = keypoint_logits.shape
    if H != W:
        raise ValueError(
            f"keypoint_logits height and width (last two elements of shape) should be equal. Instead got H = {H} and W = {W}"
        )
    discretization_size = H
    heatmaps = []
    valid = []
    for proposals_per_image, gt_kp_in_image, midx in zip(proposals, gt_keypoints, keypoint_matched_idxs):
        kp = gt_kp_in_image[midx]
        heatmaps_per_image, valid_per_image = keypoints_to_heatmap(kp, proposals_per_image, discretization_size)
        heatmaps.append(heatmaps_per_image.view(-1))
        valid.append(valid_per_image.view(-1))

    keypoint_targets = torch.cat(heatmaps, dim=0)
    valid = torch.cat(valid, dim=0).to(dtype=torch.uint8)
    valid = torch.where(valid)[0]

    # torch.mean (in binary_cross_entropy_with_logits) doesn't
    # accept empty tensors, so handle it sepaartely
    if keypoint_targets.numel() == 0 or len(valid) == 0:
        return keypoint_logits.sum() * 0

    keypoint_logits = keypoint_logits.view(N * K, H * W)

    if(train_det):
        keypoint_loss = F.cross_entropy(keypoint_logits[valid], keypoint_targets[valid], label_smoothing=0.1)
    else:
        keypoint_loss = F.cross_entropy(keypoint_logits[valid], keypoint_targets[valid])
          
    return keypoint_loss


def keypoints_to_heatmap(keypoints, rois, heatmap_size):
    # type: (Tensor, Tensor, int) -> Tuple[Tensor, Tensor]
    offset_x = rois[:, 0]
    offset_y = rois[:, 1]
    scale_x = heatmap_size / (rois[:, 2] - rois[:, 0])
    scale_y = heatmap_size / (rois[:, 3] - rois[:, 1])

    offset_x = offset_x[:, None]
    offset_y = offset_y[:, None]
    scale_x = scale_x[:, None]
    scale_y = scale_y[:, None]

    x = keypoints[..., 0]
    y = keypoints[..., 1]

    x_boundary_inds = x == rois[:, 2][:, None]
    y_boundary_inds = y == rois[:, 3][:, None]

    x = (x - offset_x) * scale_x
    x = x.floor().long()
    y = (y - offset_y) * scale_y
    y = y.floor().long()

    x[x_boundary_inds] = heatmap_size - 1
    y[y_boundary_inds] = heatmap_size - 1

    valid_loc = (x >= 0) & (y >= 0) & (x < heatmap_size) & (y < heatmap_size)
    vis = keypoints[..., 2] > 0
    valid = (valid_loc & vis).long()

    lin_ind = y * heatmap_size + x
    heatmaps = lin_ind * valid

    return heatmaps, valid
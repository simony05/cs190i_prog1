import torch
import torch.nn as nn
from utils_split9_box4 import intersection_over_union

class YOLOv1_loss(nn.Module):
    def __init__(self, split_size = 9, num_boxes = 4, num_classes = 20):
        super(YOLOv1_loss, self).__init__()
        self.mse = nn.MSELoss(reduction = "sum")
        self.split_size = split_size
        self.num_boxes = num_boxes
        self.num_classes = num_classes

        self.lambda_noobj = 0.5
        self.lambda_coord = 5

    def forward(self, predictions, target):
        predictions = predictions.reshape(-1, self.split_size, self.split_size, self.num_classes + self.num_boxes * 5)

        # IoU for 4 bounding boxes
        iou_b1 = intersection_over_union(predictions[..., 21:25], target[..., 21:25])
        iou_b2 = intersection_over_union(predictions[..., 26:30], target[..., 21:25])
        iou_b3 = intersection_over_union(predictions[..., 31:35], target[..., 21:25])
        iou_b4 = intersection_over_union(predictions[..., 36:40], target[..., 21:25])
        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0), iou_b3.unsqueeze(0), iou_b4.unsqueeze(0)], dim=0)

        iou_maxes, bestbox = torch.max(ious, dim=0)
        exists_box = target[..., 20].unsqueeze(3)

        # select predicted box coords based on bestbox (no in-place ops)
        box_predictions = (
            bestbox.eq(0).float() * predictions[..., 21:25] +
            bestbox.eq(1).float() * predictions[..., 26:30] +
            bestbox.eq(2).float() * predictions[..., 31:35] +
            bestbox.eq(3).float() * predictions[..., 36:40]
        )
        box_targets = target[..., 21:25]

        # replace in-place sqrt with out-of-place ops
        box_predictions_xy = box_predictions[..., :2]
        box_predictions_wh = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(
            torch.abs(box_predictions[..., 2:4]) + 1e-6
        )
        box_predictions = torch.cat([box_predictions_xy, box_predictions_wh], dim=-1)

        box_targets_xy = box_targets[..., :2]
        box_targets_wh = torch.sqrt(box_targets[..., 2:4] + 1e-6)
        box_targets = torch.cat([box_targets_xy, box_targets_wh], dim=-1)

        box_loss = self.mse(
            torch.flatten(exists_box * box_predictions, end_dim=-2),
            torch.flatten(exists_box * box_targets, end_dim=-2),
        )

        # object score
        pred_conf = (
            bestbox.eq(0).float() * predictions[..., 20:21] +
            bestbox.eq(1).float() * predictions[..., 25:26] +
            bestbox.eq(2).float() * predictions[..., 30:31] +
            bestbox.eq(3).float() * predictions[..., 35:36]
        )

        object_loss = self.mse(
            torch.flatten(exists_box * pred_conf),
            torch.flatten(exists_box * target[..., 20:21]),
        )

        # no object loss
        no_object_loss = 0
        for i in range(4):
            no_conf = predictions[..., 20 + i * 5 : 21 + i * 5]
            no_object_loss += self.mse(
                torch.flatten((1 - exists_box) * no_conf, start_dim=1),
                torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1),
            )

        # class prediction loss
        class_loss = self.mse(
            torch.flatten(exists_box * predictions[..., :20], end_dim=-2),
            torch.flatten(exists_box * target[..., :20], end_dim=-2),
        )

        # final loss
        total_loss = (
            self.lambda_coord * box_loss +
            object_loss +
            self.lambda_noobj * no_object_loss +
            class_loss
        )

        return total_loss

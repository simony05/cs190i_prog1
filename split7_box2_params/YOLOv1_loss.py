import torch
import torch.nn as nn
from utils import intersection_over_union

class YOLOv1_loss(nn.Module):
    def __init__(self, split_size = 7, num_boxes = 2, num_classes = 20):
        super(YOLOv1_loss, self).__init__()
        self.mse = nn.MSELoss(reduction = "sum")

        self.split_size = split_size
        self.num_boxes = num_boxes
        self.num_classes = num_classes

        self.lambda_noobj = 0.5 # no object in cell
        self.lambda_coord = 5 # bounding box coordinates

    def forward(self, predictions, target):
        predictions = predictions.reshape(-1, self.split_size, self.split_size, self.num_classes + self.num_boxes * 5)

        # IoU for 2 bounding boxes
        iou_b1 = intersection_over_union(predictions[..., 21:25], target[..., 21:25])
        iou_b2 = intersection_over_union(predictions[..., 26:30], target[..., 21:25])
        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim = 0)

        # choose bounding box with higher IoU
        iou_maxes, bestbox = torch.max(ious, dim = 0)
        exists_box = target[..., 20].unsqueeze(3)

        # box coordinates
        # set boxes with no object in them to 0
        box_predictions = exists_box * (
            (
                bestbox * predictions[..., 26:30]
                + (1 - bestbox) * predictions[..., 21:25]
            )
        )

        box_targets = exists_box * target[..., 21:25]

        box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(
            torch.abs(box_predictions[..., 2:4] + 1e-6)
        )
        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])

        box_loss = self.mse(
            torch.flatten(box_predictions, end_dim=-2),
            torch.flatten(box_targets, end_dim=-2),
        )

        # object loss
        # pred_box is the confidence score for the bounding box box with highest IoU
        pred_box = (
            bestbox * predictions[..., 25:26] + (1 - bestbox) * predictions[..., 20:21]
        )

        object_loss = self.mse(
            torch.flatten(exists_box * pred_box),
            torch.flatten(exists_box * target[..., 20:21]),
        )

        # no object loss
        no_object_loss = self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 20:21], start_dim = 1),
            torch.flatten((1 - exists_box) * target[..., 20:21], start_dim = 1),
        )

        no_object_loss += self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 25:26], start_dim = 1),
            torch.flatten((1 - exists_box) * target[..., 20:21], start_dim = 1)
        )

        # class loss
        class_loss = self.mse(
            torch.flatten(exists_box * predictions[..., :20], end_dim = -2,),
            torch.flatten(exists_box * target[..., :20], end_dim = -2,),
        )

        loss = (
            self.lambda_coord * box_loss  # first two rows in paper
            + object_loss  # third row in paper
            + self.lambda_noobj * no_object_loss  # forth row
            + class_loss  # fifth row
        )

        return loss
import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.transforms.functional as FT
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from torch.utils.data import DataLoader
from YOLOv1_model import YOLOv1_model
from dataset import VOCDataset
import time
import matplotlib.pyplot as plt
from utils import (
    non_max_suppression,
    mean_average_precision,
    intersection_over_union,
    cellboxes_to_boxes,
    get_bboxes
)
from YOLOv1_loss import YOLOv1_loss

seed = 123
torch.manual_seed(seed)

# hyperparameters
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available else "cpu"
BATCH_SIZE = 32
WEIGHT_DECAY = 5e-4
EPOCHS = 100
NUM_WORKERS = 2
PIN_MEMORY = True
IMG_DIR = "../dataset/images"
LABEL_DIR = "../dataset/labels"


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bboxes):
        for t in self.transforms:
            img, bboxes = t(img), bboxes

        return img, bboxes


transform = Compose([transforms.Resize((448, 448)), transforms.ToTensor(),])


def train_fn(train_loader, model, optimizer, loss_fn):
    loop = tqdm(train_loader, leave = True)
    mean_loss = []

    for batch_idx, (x, y) in enumerate(loop):
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        loss = loss_fn(out, y)
        mean_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update progress bar
        loop.set_postfix(loss = loss.item())

    print(f" Mean loss was {sum(mean_loss)/len(mean_loss)}")

    return sum(mean_loss)/len(mean_loss)


def main():
    model = YOLOv1_model(split_size = 7, num_boxes = 2, num_classes = 20).to(DEVICE)
    optimizer = optim.Adam(
        model.parameters(), lr = LEARNING_RATE, weight_decay = WEIGHT_DECAY
    )
    scheduler = StepLR(optimizer, step_size = 30, gamma = 0.1)
    loss_fn = YOLOv1_loss()

    train_dataset = VOCDataset(
        # "./dataset/100examples.csv",
        "../dataset/train.csv",
        transform = transform,
        img_dir = IMG_DIR,
        label_dir = LABEL_DIR,
    )

    test_dataset = VOCDataset(
        "../dataset/test.csv", 
        transform = transform, 
        img_dir = IMG_DIR, 
        label_dir = LABEL_DIR,
    )

    train_loader = DataLoader(
        dataset = train_dataset,
        batch_size = BATCH_SIZE,
        num_workers = NUM_WORKERS,
        pin_memory = PIN_MEMORY,
        shuffle = True,
        drop_last = True,
    )

    test_loader = DataLoader(
        dataset = test_dataset,
        batch_size = BATCH_SIZE,
        num_workers = NUM_WORKERS,
        pin_memory = PIN_MEMORY,
        shuffle = True,
        drop_last = True,
    )

    map_scores = []
    losses = []
    total_start = time.time()

    for epoch in range(EPOCHS):

        pred_boxes, target_boxes = get_bboxes(
            train_loader, model, iou_threshold = 0.5, threshold = 0.4
        )

        mean_avg_prec = mean_average_precision(
            pred_boxes, target_boxes, iou_threshold = 0.5, box_format = "midpoint"
        )

        print(f"Epoch {epoch} Train mAP: {mean_avg_prec}")
        map_scores.append(mean_avg_prec)

        loss = train_fn(train_loader, model, optimizer, loss_fn)

        scheduler.step()

        losses.append(loss)

        with open("map_scores.txt", "a") as f:
            f.write(f"{mean_avg_prec}\n")

        with open("avg_loss.txt", "a") as f:
            f.write(f"{loss}\n")

    total_end = time.time()
    total_time = total_end - total_start
    avg_epoch_time = total_time / EPOCHS

    print(f"\nTotal training time: {total_time:.2f} seconds")
    print(f"Average epoch time: {avg_epoch_time:.2f} seconds")

    plt.figure()
    plt.plot(map_scores, linestyle='-')
    plt.title("YOLOv1 Training mAP per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("mAP")
    plt.tight_layout()
    plt.savefig("mAP.png")
    plt.show()

    plt.figure()
    plt.plot(losses, linestyle='-')
    plt.title("YOLOv1 Training Mean Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Loss")
    plt.tight_layout()
    plt.savefig("loss.png")
    plt.show()



if __name__ == "__main__":
    main()

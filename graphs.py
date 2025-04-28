import matplotlib.pyplot as plt
import numpy as np

y1 = np.loadtxt('./values/avg_loss_split7_box2.txt')
y2 = np.loadtxt('./values/avg_loss_split9_box4.txt')

epochs = np.arange(len(y1))
plt.figure()
plt.plot(epochs, y1, label='7x7, 2 bounding boxes', linestyle='-')
plt.plot(epochs, y2, label='9x9, 4 bounding boxes', linestyle='-')

plt.title("YOLOv1 Mean Loss per Epoch (Constant LR)")
plt.xlabel("Epoch")
plt.ylabel("Mean Loss")
plt.legend()
plt.tight_layout()
plt.savefig("meanloss_constLR.png")
plt.show()
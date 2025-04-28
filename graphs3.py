import matplotlib.pyplot as plt
import numpy as np

y1 = np.loadtxt('./values/avg_loss72.txt')
y2 = np.loadtxt('./values/avg_loss72_32.txt')
y3 = np.loadtxt('./values/avg_lossG.txt')

y1 = y1[:80]
y2 = y2[:80]
y3 = y3[:80]

epochs = np.arange(len(y1))
plt.figure()
plt.plot(epochs, y1, label='Batchnorm 16', linestyle='-')
plt.plot(epochs, y2, label='Batchnorm 32', linestyle='-')
plt.plot(epochs, y3, label='Groupnorm 16', linestyle='-')

plt.title("YOLOv1 Mean Loss per Epoch (Scheduler)")
plt.xlabel("Epoch")
plt.ylabel("Mean Loss")
plt.legend()
plt.tight_layout()
plt.savefig("mean_loss_scheduler.png")
plt.show()
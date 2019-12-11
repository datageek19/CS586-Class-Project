import matplotlib.pyplot as plt
import numpy as np

tf_train_loss = {}
tf_test_loss = {}
with open('size-20M.log', 'r') as file:
    iter = 1
    i = 1
    for line in file:
        if i % 2 == 1:
            line = line.split('|')[1]
            line = float(line.replace("Train Loss: ", ""))
            tf_train_loss[iter] = line
        else:
            line = line.split('|')[1]
            line = float(line.replace("Test  Loss: ", ""))
            tf_test_loss[iter] = line
            iter += 1
        i += 1

plt.subplot(1, 2, 1)
plt.grid()
plt.plot(epochs, cs_loss, '-', color= (0, 0, 0, 1),linewidth=2, label='C#')
plt.plot(epochs, tf_train_loss, 'b-', linewidth=2, label='TF-Small')
plt.plot(epochs, tf_train_loss_large20, 'r-', linewidth=2, label='TF-Large')
plt.plot(epochs, tf_train_loss_large20_dense, 'g-', linewidth=2, label='TF-Large(Reduced)')

plt.plot(epochs, tf_test_loss, 'b--', linewidth=2)
plt.plot(epochs, tf_test_loss_large20, 'r--', linewidth=2)
plt.plot(epochs, tf_test_loss_large20_dense, 'g--', linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.grid(axis='y')
barwidth = 0.3
COUNT = 4
x = np.arange(COUNT) + 1 - barwidth / 2
time = [50, 28.75, 660/18.0, 676.49/18.0]

barlist = plt.bar(x, time, barwidth)
barlist[0].set_color((0, 0, 0, 0.8))
barlist[1].set_color('b')
barlist[2].set_color('r')
barlist[3].set_color('g')

plt.xticks(x + barwidth / 2, ('C#', 'TF-Small', 'TF-Large', 'TF-Large(Reduced)'))
plt.title('Train Time')
plt.ylabel("Training Time Per 1M Queries (s)")
plt.xlim(0.5, COUNT + 0.5)
plt.ylim(0, 60)

plt.tight_layout()
plt.show()
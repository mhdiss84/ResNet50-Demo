import matplotlib.pyplot as plt
import re

epoch_losses = {}

with open('train.log') as f:
    for line in f:
        match = re.search(r'Epoch \[(\d+)\] Step .* Loss: ([\d.]+)', line)

        if match:
            epoch = int(match.group(1))
            loss = float(match.group(2))
            if epoch not in epoch_losses:
                epoch_losses[epoch].append(loss)

epochs = sorted(epoch_losses.keys())
avg_losses = [sum(epoch_losses[e]) / len(epoch_losses[e]) for e in epochs]

plt.figure(figsize=(10, 5))
plt.plot(epochs, avg_losses, marker='o')
plt.title('Training Loss per Epoch (averages)')
plt.xlabel('Epoch')
plt.ylabel('Average Loss')
plt.grid(True)
plt.savefig('loss_curve1.png')
print('Saved loss_curve1.png')
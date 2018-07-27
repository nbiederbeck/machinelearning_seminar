import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

history = pd.read_pickle('history.pkl')
print(history.keys())

epochs = len(history)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(range(epochs), history.val_loss, 'r-.', label='validation loss')
ax.plot(range(epochs), history.loss, 'b-.', label='training loss')
ax.set_xlabel('Epoche')
ax.set_ylabel('loss')

ax = ax.twinx()
ax.plot(range(epochs), history.val_acc, 'r-', label='validation acc')
ax.plot(range(epochs), history.acc, 'b-', label='training acc')
ax.set_ylabel('Accuracy')

ax.legend(loc='best')
plt.show()


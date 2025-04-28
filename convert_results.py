import pandas as pd
import matplotlib.pyplot as plt

# Load results
LOCAL_PATH = "/work/imborhau/"
df = pd.read_csv(LOCAL_PATH + 'runs/detect/train66/results.csv')

# Create figure
fig, axs = plt.subplots(2, 2, figsize=(14, 10))
axs = axs.ravel()

# Plot Training Loss
axs[0].plot(df['epoch'], df['train/box_loss'], label='Box Loss', color='royalblue')
axs[0].plot(df['epoch'], df['train/cls_loss'], label='Class Loss', color='seagreen')
axs[0].set_title('Training Loss', fontsize=16)
axs[0].set_xlabel('Epoch', fontsize=12)
axs[0].set_ylabel('Loss', fontsize=12)
axs[0].legend()
axs[0].grid(True)

# Plot Precision
axs[1].plot(df['epoch'], df['metrics/precision(B)'], label='Precision', color='darkorange')
axs[1].set_title('Precision', fontsize=16)
axs[1].set_xlabel('Epoch', fontsize=12)
axs[1].set_ylabel('Precision', fontsize=12)
axs[1].legend()
axs[1].grid(True)

# Plot Recall
axs[2].plot(df['epoch'], df['metrics/recall(B)'], label='Recall', color='firebrick')
axs[2].set_title('Recall', fontsize=16)
axs[2].set_xlabel('Epoch', fontsize=12)
axs[2].set_ylabel('Recall', fontsize=12)
axs[2].legend()
axs[2].grid(True)

# Plot mAP50
axs[3].plot(df['epoch'], df['metrics/mAP50(B)'], label='mAP50', color='mediumvioletred')
axs[3].set_title('mAP50', fontsize=16)
axs[3].set_xlabel('Epoch', fontsize=12)
axs[3].set_ylabel('mAP@0.5', fontsize=12)
axs[3].legend()
axs[3].grid(True)

# Layout
plt.tight_layout(pad=3.0)
plt.suptitle('Training Metrics Overview', fontsize=20, y=1.05)
plt.subplots_adjust(top=0.92)

# Save nicely
plt.savefig('pretty_results.png', dpi=300)
plt.show()

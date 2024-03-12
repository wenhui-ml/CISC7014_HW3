import matplotlib.pyplot as plt
import numpy as np

# Model names
models = ['ResNet18', 'ResNet50', 'MLP3Layer', 'Swin-Transformer-Tiny']

# Class names
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Define the x-axis values
x = np.arange(len(classes))

# Define the width of the bars
width = 0.2

# Assume you have number of images per class for normal, long-tail and weighted loss scenarios
normal_images = [5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000]
longtail_images = [5000, 4500, 4000, 3500, 3000, 2500, 2000, 1500, 1000, 500]
weighted_images = [5000, 4500, 4000, 3500, 3000, 2500, 2000, 1500, 1000, 500]

# Per-class accuracy for each model
per_class_accuracies = [
    # Normal CIFAR-10
    [
        [96, 96, 93, 89, 95, 90, 97, 96, 96, 95], # ResNet18
        [98, 97, 95, 92, 95, 93, 98, 97, 97, 96], # ResNet50
        [54, 42, 17, 66, 20, 14, 45, 41, 52, 57], # MLP3Layer
        [97, 98, 97, 93, 97, 94, 99, 97, 98, 96]  # Swin-Transformer-Tiny
    ],
    # Long-tailed CIFAR-10
    [
        [98, 99, 93, 88, 95, 84, 94, 87, 82, 73], # ResNet18
        [99, 98, 95, 92, 96, 89, 96, 90, 86, 75], # ResNet50
        [72, 68, 47, 40, 33,  5, 33, 41, 26, 15], # MLP3Layer
        [99, 99, 96, 92, 97, 91, 97, 92, 88, 78]  # Swin-Transformer-Tiny
    ],
    # Long-tailed CIFAR-10 with class-weighted loss
    [
        [97, 98, 93, 90, 94, 83, 94, 87, 85, 76], # ResNet18
        [99, 98, 96, 92, 96, 91, 93, 93, 86, 77], # ResNet50
        [23, 45, 18,  0, 10, 60, 28, 42, 23, 48], # MLP3Layer
        [99, 99, 97, 93, 97, 91, 95, 93, 89, 81]  # Swin-Transformer-Tiny
    ]
]

for model_type, accuracies in zip(['Normal CIFAR-10', 'Long-tailed CIFAR-10', 'Long-tailed CIFAR-10 (Weighted Loss)'], per_class_accuracies):
    fig, ax1 = plt.subplots(figsize=(15, 7))

    color = 'tab:blue'
    ax1.set_xlabel('Class')
    ax1.set_ylabel('Per-Class Accuracy', color=color)
    for i, model in enumerate(models):
        bars = ax1.bar(x - width + i * width, accuracies[i], width, label=model, color=plt.cm.Paired(i / len(models))) # added label here
        for bar in bars:
            yval = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2.0, yval, round(yval, 2), va='bottom', ha='center')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Number of Images', color=color)
    lines = [ax2.plot(x, normal_images, color='b', marker='o', label='Normal CIFAR-10 Images'),
             ax2.plot(x, longtail_images, color='g', marker='s', label='Long-tailed CIFAR-10 Images'),
             ax2.plot(x, weighted_images, color='r', marker='^', label='Long-tailed CIFAR-10 with Weighted Loss Images')]
    for line in lines:
        for i, txt in enumerate(line[0].get_ydata()):
            ax2.annotate(txt, (x[i], txt))
    ax2.tick_params(axis='y', labelcolor=color)

    # Add class names to the plot
    for i in x:
        ax1.text(i, -5, classes[i], ha='center', va='top')

    # Add model names to the plot
    for i, model in enumerate(models):
        ax1.text(9.5, 10 + i*5, model, ha='left', va='center', color=plt.cm.Paired(i / len(models)), 
                bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5', alpha=0.5))

    fig.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.title(f'Per-Class Accuracy for {model_type}')
    plt.legend() # removed models here
    plt.savefig(f'per_class_accuracy_{model_type.replace(" ", "_")}.png')

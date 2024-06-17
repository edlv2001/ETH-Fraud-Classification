import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import numpy as np

true_negative = 2804
false_positive = 122
false_negative = 85
true_positive = 941

# Calculate the metrics
accuracy = (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)
precision = true_positive / (true_positive + false_positive)
recall = true_positive / (true_positive + false_negative)
f1_score = 2 * (precision * recall) / (precision + recall)

accuracy, precision, recall, f1_score
# Simulated data for ROC curve
y_true = np.concatenate([np.zeros(true_negative + false_positive), np.ones(false_negative + true_positive)])
y_scores = np.concatenate([np.zeros(true_negative), np.ones(false_positive), np.zeros(false_negative), np.ones(true_positive)])

# Calculate the ROC curve
fpr, tpr, _ = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

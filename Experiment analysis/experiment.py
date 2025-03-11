import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

reactions = np.array([188, 267, 217, 297, 267, 263, 267,
                     316, 253, 311, 250, 328, 193, 273, 232])
lengths = np.array([0.3, 0.4875, 0.425, 0.55, 0.3, 0.425, 0.425,
                   1.05, 0.425, 0.675, 0.4875, 0.675, 0.425, 0.675, 0.3])

# Fit a linear regression model
p = np.polyfit(reactions, lengths, 1)  # fit degree 1 polynomial to data
r = np.linspace(np.min(reactions), np.max(reactions), 100)
fits = p[0] * r + p[1]  # generate points from the fit to plot

# Calculate correlation coefficient and p-value
correlation_coefficient, p_value = pearsonr(reactions, lengths)
print(f"Correlation Coefficient: {correlation_coefficient:.4f}")
print(f"P-value: {p_value:.4f}")

# Plot data and fit line
plt.scatter(reactions, lengths, label='Datapoints')
plt.plot(r, fits, color='red', label='Line of best fit')
plt.title("Critical Length vs. Reaction Time from Experiment")
plt.xlabel("Reaction Time (ms)")
plt.ylabel("Critical Length (m)")
plt.legend()
plt.savefig("experimentPlot.svg")
plt.grid()
plt.show()

import matplotlib.pyplot as plt
import numpy as np
reactions = np.array([188,267,217,297,267,263,267,316,253,311,250,328,193,273,232])
lengths = np.array([0.3,0.4875,0.425,0.55,0.3,0.425,0.425,1.05,0.425,0.675,0.4875,0.675,0.425,0.675,0.3])

p = np.polyfit(reactions, lengths,1) #fit degree 1 polynomial to data
r = np.linspace(np.min(reactions),np.max(reactions), 100) #
fits = np.array([(p[0]*i + p[1]) for i in r]) #generate points from the fit to plot

plt.scatter(reactions, lengths)
plt.plot(r, fits)
#scatter plot the data, plot the fit line
plt.title("Reaction Time against Critical Length from experimental data")
plt.xlabel("reaction time (ms)")
plt.ylabel("critical length (m)")
plt.savefig("experimentPlot.svg")
plt.show()
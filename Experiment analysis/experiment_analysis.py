import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

test_data = pd.read_csv('<name of csv>')
# print(test_data.head())

lengths = np.array(test_data[:0])
# print(lengths)

# for index, row in test_data.iterrows():
#     print(index, row["Reaction time avg (ms)"])

test_data["Critcal length "] = test_data["Critcal length "].astype(
    str).str.extract(r'(\d+\.?\d*)').astype(float)


# Array of all reaction times
rts = np.array(test_data["Reaction time avg (ms)"])
# Array of all critcal lengths
critical_lengths = np.array(test_data["Critcal length "])
# Average reaction time
meanrt = test_data["Reaction time avg (ms)"].mean()


plt.figure(figsize=(8, 6))
plt.scatter(critical_lengths, rts, color="blue",
            label="Data Points", alpha=0.7)
plt.axhline(meanrt, color="red", linestyle="--",
            label=f"Mean RT: {meanrt:.2f} ms")

plt.xlabel("Critical Length")
plt.ylabel("Reaction Time (ms)")
plt.title("Reaction Time vs. Critical Length")
plt.legend()
plt.grid(True)

plt.show()

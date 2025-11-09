# prog_sp_02-3.py
import numpy as np
import matplotlib.pyplot as plt
# import japanize_matplotlib # If using Japanese labels

# Set the number of samples
num_samples = 1000

# Sample X from a uniform distribution over the interval [-1, 1]
x_samples = np.random.uniform(-1, 1, num_samples)
y_samples_uni = np.random.uniform(-1, 1, num_samples)

# generate Y of either (1),(2),(3).
#y_samples = x_samples      #(1) positive correlated (Y=X)
#y_samples = y_samples_uni  #(2) uncorrelated and independent
y_samples = x_samples**2    #(3) uncorrelated but not independent (Y = X^2))

# add random noises
d_samples = np.random.uniform(-0.4, 0.4, num_samples)
y_samples = y_samples + d_samples

# Calculate the covariance of the sample data (should theoretically be close to 0)
covariance = np.cov(x_samples, y_samples)[0, 1]
correlation = np.corrcoef(x_samples, y_samples)[0, 1]

print(f"Number of Samples: {num_samples}")
print(f"Sample Covariance: {covariance:.4f}")
print(f"Sample Correlation Coefficient: {correlation:.4f}")


# Plot the data
plt.figure(figsize=(8, 6))
plt.scatter(x_samples, y_samples, alpha=0.6, label=f'Sample Points (N={num_samples})')

# plt.title('Example of Positively Correlated Random Variables')                      #(1)
# plt.title('Example of Uncorrelated and Independent Random Variables')               #(2)
plt.title('Example of Uncorrelated but Not Independent Random Variables ($Y = X^2$)') #(3)

plt.xlabel('X Sampled from U[-1, 1]')

# plt.ylabel('Y Sampled from X + noise') #(1)
# plt.ylabel('Y Sampled from U[-1, 1]')  #(2)
plt.ylabel('Y Sampled from X^2 + noise') #(3)

plt.axhline(0, color='grey', lw=0.5)
plt.axvline(0, color='grey', lw=0.5)
plt.grid(True)
plt.legend()
plt.text(-0.5, 0.9, f'Sample Covariance: {covariance:.3f}\nSample Correlation Coef: {correlation:.3f}',
          bbox=dict(boxstyle='round,pad=0.3', fc='wheat', alpha=0.5))

#plt.savefig("./sp2-3_positively_correlated.png")       #(1)
#plt.savefig("./sp2-3_uncorrelated_independent.png")    #(2)
plt.savefig("./sp2-3_uncorrelated_not_independent.png") #(3)

plt.show()

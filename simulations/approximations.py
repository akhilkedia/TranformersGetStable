import numpy as np
import matplotlib
# matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

x = np.linspace(start=0, stop=1, num=5000)
sigma = (np.pi - 1) / (2 * np.pi)
cov_relu = (1 / 4 + np.arcsin(x) / (2 * np.pi)) * x - (1 - (1 - x**2)**0.5) / (2 * np.pi)
cor_relu = cov_relu / sigma
cor_relu_approx = 0.7 * x + 0.3 * x * x

plt.title('relu correlation forward')
plt.plot(x, x, label='Input Correlation')
plt.plot(x, cor_relu, label='Full Relu Correlation')
plt.plot(x, cor_relu_approx, label='Approximate Relu Correlation')
plt.legend()
plt.show()


corr_ffn = (1 / 2 + np.arcsin(x) / np.pi) * x + (1 - x**2)**0.5 / np.pi
cor_relu_approx = 1/np.pi + 0.5*x + (0.5 - 1/np.pi)*x*x
plt.title('FFN correlation forward')
plt.plot(x, corr_ffn, label='Full Relu Correlation')
plt.plot(x, cor_relu_approx, label='Approximate Relu Correlation')
plt.legend()
plt.show()




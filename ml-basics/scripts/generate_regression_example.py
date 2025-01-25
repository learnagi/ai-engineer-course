import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# Generate sample data
X = 2 * np.random.rand(100, 1)  # Generate 100 random x values
y = 4 + 3 * X + np.random.randn(100, 1)  # Generate corresponding y values with noise

# Create figure
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', alpha=0.5, label='Data Points')
plt.xlabel('Feature X (e.g., House Area)')
plt.ylabel('Target y (e.g., Price)')
plt.title('Linear Regression Example')

# Add two "guessed" lines
plt.plot([0, 2], [4, 10], 'r--', label='Possible Fit Line 1')
plt.plot([0, 2], [4, 10.5], 'g--', label='Possible Fit Line 2')

# Add legend
plt.legend()

# Save figure
plt.savefig('images/linear-regression-example.png', dpi=300, bbox_inches='tight')
plt.close()

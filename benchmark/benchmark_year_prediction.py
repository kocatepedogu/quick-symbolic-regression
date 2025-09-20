import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from quicksr import *

# Load the dataset into a pandas DataFrame
df = pd.read_csv('dataset/YearPredictionMSD.csv')

# Convert the DataFrame to a numpy array
data_array = df.to_numpy()

# Separate features and target
target = data_array[:, 0]
features = data_array[:, 1:]

# Set model hyperparameters
NVARS=features.shape[1]
NPOPULATION=35
NISLANDS=1
NWEIGHTS=2

# Create model
model = SymbolicRegressionModel(NVARS, NWEIGHTS, NPOPULATION, NISLANDS,
                                runner_generator=IntraIndividualRunnerGenerator())

# Fit model
solution, loss_wrt_gen, loss_wrt_time, time_hist = model.fit(features, target, ngenerations=1, nsupergenerations=50, nepochs=2)

# Print best solution
print("Best solution: {}".format(solution))

# Print best loss as RMSE
print("Best loss (RMSE): {}".format(np.sqrt(solution.loss)))

# Plot learning history
plt.figure(figsize=(10, 6))
plt.plot(time_hist, loss_wrt_time)
plt.xlabel('Time (milliseconds)')
plt.ylabel('Loss (MSE)')
plt.title('Learning History')
plt.grid()
plt.savefig('benchmark_learning_history_year_prediction.png')

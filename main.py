
# Importing required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from keras.layers import Input, Dense
from keras.models import Model
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest


# In[95]:


# Loading sensor data
train_data = pd.read_csv('train_FD003.txt', delimiter=' ', header=None)
train_data.drop(train_data.columns[[26, 27]], axis=1, inplace=True)
train_data.drop(train_data.columns[[0, 1]], axis=1, inplace=True)

# Data pre-processing
scaler = StandardScaler()
train_data_scaled = scaler.fit_transform(train_data)

# In[96]:


train_data

# Autoencoder model architecture
input_layer = Input(shape=(train_data_scaled.shape[1],))
encoded = Dense(64, activation='relu')(input_layer)
encoded = Dense(32, activation='relu')(encoded)
encoded = Dense(16, activation='relu')(encoded)
decoded = Dense(32, activation='relu')(encoded)
decoded = Dense(64, activation='relu')(decoded)
decoded = Dense(train_data_scaled.shape[1], activation=None)(decoded)

autoencoder = Model(input_layer, decoded)
autoencoder.compile(optimizer='adam', loss='mse')

# Model training
autoencoder.fit(train_data_scaled, train_data_scaled, epochs=20, batch_size=32, validation_split=0.2)


# In[116]:


# Loading test data
test_data = pd.read_csv('test_FD003.txt', delimiter=' ', header=None)
test_data.drop(test_data.columns[[26, 27]], axis=1, inplace=True)
test_data.drop(test_data.columns[[0, 1]], axis=1, inplace=True)

# Data pre-processing
test_data_scaled = scaler.transform(test_data)


# In[117]:


# Prediction and reconstruction error calculation
reconstructed_data = autoencoder.predict(test_data_scaled)
mse = np.mean(np.power(test_data_scaled - reconstructed_data, 2), axis=1)


# In[123]:


# Plotting the reconstruction error as a scatter plot
plt.scatter(range(len(mse)), mse, c='blue', alpha=0.5)
plt.plot([0.3, len(mse)], [threshold, threshold], c='red')
plt.xlabel('Sample')
plt.ylabel('Reconstruction error')
plt.show()


# In[124]:


# Calculating the threshold for anomaly detection
threshold = np.max(mse) * 0.8

# Identifying anomalies
anomalies = np.where(mse > threshold)[0]
print('Number of anomalies:', len(anomalies))


# In[125]:


# Get the actual anomalies from the test data
actual_anomalies = np.where(test_data.iloc[:, -1] <= 0)[0]
print('Number of actual anomalies:', len(actual_anomalies))


# In[126]:


# Evaluation metrics
accuracy = 1 - len(anomalies) / len(test_data)

print('Accuracy:', accuracy)


# In[127]:


plt.scatter(range(len(mse)), mse, c=['blue' if e < threshold else 'red' for e in mse])
plt.axhline(y=threshold, color='r', linestyle='-')
plt.xlabel('Data point')
plt.ylabel('Reconstruction error')
plt.title('Autoencoder Anomaly Detection')
plt.show()


# # Isolation Forest Model

# In[128]:


# Train Isolation Forest model
clf = IsolationForest(n_estimators=100, max_samples='auto', contamination='auto', random_state=42)
clf.fit(train_data)

# Load test data
test_data = pd.read_csv('test_FD003.txt', delimiter=' ', header=None)
test_data.drop(test_data.columns[[26, 27]], axis=1, inplace=True)
test_data.drop(test_data.columns[[0, 1]], axis=1, inplace=True)

# Predict the anomaly scores for test data
scores = clf.score_samples(test_data)


# In[145]:


# Calculate the threshold for anomaly detection
threshold = np.percentile(scores, 1)

# Identify anomalies in test data
anomalies = np.where(scores < threshold)[0]

# Print the number of anomalies detected
print('Number of anomalies:', len(anomalies))


# In[146]:


# Plot the distribution of anomaly scores
plt.hist(scores, bins=50, color='b')
plt.axvline(threshold, color='r', linestyle='dashed', linewidth=2)
plt.title('Anomaly Scores Distribution')
plt.xlabel('Anomaly Score')
plt.ylabel('Count')
plt.show()





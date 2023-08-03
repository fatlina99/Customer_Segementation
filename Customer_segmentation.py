#%%
# Import libraries
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.utils import plot_model
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report


import pickle
import os, datetime
# %%
# 1. Data Preprocessing 
df = pd.read_csv('customer_segmentation.csv')
df_test = pd.read_csv('new_customers.csv')
# %%
df.head()
# %%
# 2. Data Cleaning

df.isnull().sum()

#%%
df_test.isnull().sum()

#%%
df.head(30)
# %%
# Fill in categorical columns missing value with a 'unknown' 
df_cat = ['Ever_Married', 'Profession', 'Graduated', 'Var_1', 'Gender', 'Spending_Score']
df_test_cat = ['Ever_Married', 'Profession', 'Graduated', 'Var_1', 'Gender', 'Spending_Score']

df[df_cat] = df[df_cat].fillna('Unknown')
df_test[df_test_cat] = df_test[df_test_cat].fillna('Unknown')


# %%
# Impute numerical columns with the mean
df['Work_Experience'].fillna(df['Work_Experience'].mean(), inplace=True)
df_test['Work_Experience'].fillna(df['Work_Experience'].mean(), inplace=True)

# Impute numerical columns with the median
df['Family_Size'].fillna(df['Family_Size'].median(), inplace=True)
df_test['Family_Size'].fillna(df['Family_Size'].median(), inplace=True)

#%%
# Convert Categorical Columns into numerical form  
label_encoder = LabelEncoder()
for col in df_cat:
    df[col] = label_encoder.fit_transform(df[col])

df['Segmentation'] = label_encoder.fit_transform(df['Segmentation'])
# %%
# 3. Perform train-test split
X = df.drop(columns=['Segmentation', 'ID'])
y = df['Segmentation']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#%%
# 4. Data Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# %%

# 5. Deep Learning
def create_model(input_dim):
    model = Sequential()
    model.add(Dense(128, activation='relu', input_dim=input_dim))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(Dense(1, activation='sigmoid'))  # Sigmoid activation for binary classification
    return model

model = create_model(input_dim=X_train_scaled.shape[1])

# %%
# 6. Compile the model
from tensorflow.keras.optimizers import Adam 
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

early_stopping = EarlyStopping(patience=5, restore_best_weights=True)

tensorboard = TensorBoard(log_dir='./logs', histogram_freq=1)

#%%
# 7. Model Training
history = model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, validation_split=0.2, callbacks=[early_stopping, tensorboard])
# %%
# 8. Evaluation
loss, accuracy = model.evaluate(X_test_scaled, y_test)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)

#%%
# 9. Model Saving
# Save the model in .h5 format
model.save('customer_segmentation_model.h5')

# Save the scaler in .pkl format
with open('scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)

#%%
# 10. Machine Learning
# Initialize and train KNN model
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_scaled, y_train)

# Initialize and train Gradient Boosting model
gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
gb_model.fit(X_train_scaled, y_train)
#%%
# Deep Learning Model Predictions
y_pred_dl = model.predict(X_test_scaled)
y_pred_dl = (y_pred_dl > 0.5).astype(int)  # Convert probabilities to binary predictions
#%%
# KNN Model Predictions
y_pred_knn = knn_model.predict(X_test_scaled)

# Gradient Boosting Model Predictions
y_pred_gb = gb_model.predict(X_test_scaled)

#%%
# Evaluation using F1-score
dl_f1_score = f1_score(y_test, y_pred_dl, average='weighted')
knn_f1_score = f1_score(y_test, y_pred_knn, average='weighted')
gb_f1_score = f1_score(y_test, y_pred_gb, average='weighted')

print(f"Deep Learning Model F1-score: {dl_f1_score}")
print(f"KNN Model F1-score: {knn_f1_score}")
print(f"Gradient Boosting Model F1-score: {gb_f1_score}")

#%%
# Plot and Save Model Architecture
plot_model(model, to_file='model_architecture.png', show_shapes=True)

#%%
# Plot and Save F1-Score
plt.figure()
models = ['Deep Learning', 'KNN', 'Gradient Boosting']
f1_scores = [dl_f1_score, knn_f1_score, gb_f1_score]
plt.bar(models, f1_scores)
plt.xlabel('Models')
plt.ylabel('F1-Score')
plt.title('F1-Score Comparison')
plt.savefig('f1_score_comparison.png')
# %%
# Conclusion
"""
1. The Deep Learning Model has the lowest F1-score = 0.0851. This indicates that the model is not performing well on the classification task, and its predictions are not precise or recall-friendly.

2. The KNN Model has a higher F1-score =0 .4823 compared to the Deep Learning Model. However, it is still relatively low, suggesting that the KNN model is better than the Deep Learning Model but still needs improvement.

3. The Gradient Boosting Model has the highest F1-score =0.5215 among the three models. This indicates that the Gradient Boosting model is performing the best in terms of precision and recall.
"""
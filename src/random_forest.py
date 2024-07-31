import boto3
import joblib as jb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sagemaker
from sklearn.ensemble import RandomForestClassifier
import sklearn
import pathlib
from io import StringIO
import argparse
import os

sm_boto3 = boto3.client("sagemaker")
session = sagemaker.Session()
region = session.boto_session.region_name
bucket = "anomaly-detection-motor"

train_path = "../data/motor_data_train.csv"
test_path = "../data/motor_data_test.csv"

# Getting the datasets
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

# Splliting x and y in training datasets
y_train = train_df.pop("label")
x_train = train_df

# Splitting x and y in testing datasets
y_test = test_df.pop("label")
x_test = test_df

# Upload Data to S3 Bucket
sk_prefix = "sagemaker/anamoly_detection_100HP_Motor/train_test_data"
train_path = session.upload_data(
    path="../data/motor_data_train.csv", bucket=bucket, key_prefix=sk_prefix
)
test_path = session.upload_data(
    path="../data/motor_data_test.csv", bucket=bucket, key_prefix=sk_prefix
)


# ----------------------CREATING THE MODEL-------------------------#parser = argparse.ArgumentParser()
parser = argparse.ArgumentParser()
# Required Arguments
parser.add_argument("--model-dir", type=str, default=os.enviorn.get("SM_MODEL_DIR"))
parser.add_argument("--train", type=str, defa   ult=os.enviorn.get("SM_CHANNEL_TRAIN"))
parser.add_argument("--test", type=str, default=os.enviorn.get("SM_CHANNEL_TEST"))
parser.add_argument("--train-file", type=str, default="model_data_train.csv")
parser.add_argument("--test-file", type=str, default="model_data_test.csv")

# Parsing Arguments
args, _ = parser.parse_known_args()


def accuracy(y_pred, y_test):
    correct_preds = 0
    wrong_preds_idxs = []
    for i in range(0, len(y_pred)):
        if y_pred[i] == y_test[i]:
            correct_preds += 1
        else:
            wrong_preds_idxs.append(i)

    accuracy = round(correct_preds / len(y_test), 4) * 100
    print(f"{accuracy}% accuracy.")
    print(y_pred)
    print(wrong_preds_idxs)


train_df = pd.read_csv(args.train_file)


clf = RandomForestClassifier(n_estimators=100, max_depth=25)

# Fitting the training dataset
clf.fit(x_train, y_train)

# Making predictions
y_pred = clf.predict(x_test)

# Finding the accuracy of the model
accuracy(y_pred, y_test)

# -----------------------GRAPHING THE DATA--------------------------#
# -------Training-------#

# Plotting
plt.figure(figsize=(60, 8))

# Plot all in one graph
plt.plot(train_df["temperature"], label="Temperature(F)", marker="d", color="purple")
plt.plot(train_df["current"], label="Current(A)", marker="d", color="orange")
plt.plot(train_df["vibration"], label="Vibration(mm/s)", marker="d", color="green")
plt.plot(train_df["voltage"], label="Voltage(V)", marker="d", color="blue")


plt.title("Sensor Readings")
plt.xlabel("Seconds")
plt.ylabel("Values")
plt.legend()

plt.show()

# -------Testing-------#

# Plotting
plt.figure(figsize=(100, 8))
idxs = []
for i in range(0, len(y_pred)):
    if y_pred[i] == 1:
        idxs.append(i)


# Plot all in one graph
plt.plot(test_df["temperature"], label="Temperature(F)", marker="d", color="purple")
plt.plot(test_df["current"], label="Current(A)", marker="d", color="orange")
plt.plot(test_df["vibration"], label="Vibration(mm/s)", marker="d", color="green")
plt.plot(test_df["voltage"], label="Voltage(V)", marker="d", color="blue")


# Highlight specific points with red outlines
plt.scatter(
    idxs,
    test_df.loc[idxs, "temperature"],
    edgecolor="red",
    facecolor="red",
    s=100,
    label=None,
)
plt.scatter(
    idxs,
    test_df.loc[idxs, "current"],
    edgecolor="red",
    facecolor="red",
    s=100,
    label=None,
)
plt.scatter(
    idxs,
    test_df.loc[idxs, "vibration"],
    edgecolor="red",
    facecolor="red",
    s=100,
    label=None,
)
plt.scatter(
    idxs,
    test_df.loc[idxs, "voltage"],
    edgecolor="red",
    facecolor="red",
    s=100,
    label=None,
)


plt.title("Sensor Readings")
plt.xlabel("Seconds")
plt.ylabel("Values")
plt.legend(
    loc="upper left",
    bbox_to_anchor=(1, 1),
)

plt.show()

# Export model
jb.dump(clf, "../models/Anomaly_Detection_100HP_Motor.pkl")

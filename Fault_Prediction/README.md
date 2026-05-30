# Fault Prediction

## Challenge Description

Modern manufacturing relies on hundreds of highly reliable machines working together to maintain efficiency and productivity. However, even with advanced equipment, unexpected machine downtime continues to cost industries millions of dollars each year in lost output, delayed operations, and increased maintenance expenses.

This challenge invites students to design and develop intelligent systems aimed at minimizing such disruptions. Participants are encouraged to explore a wide range of solutions, including Machine Learning models, sensor-based monitoring systems, or integrated robotic and sensor arrays that can continuously track machine performance.

## Potential Solutions

- Train an Machine Learning model using robot telemetry data to detect failure
- Create a dashboard to monitor machine states
- Install basic sensors on machines to assess their status
- Design/Program testing procedures that assess machine health

## Recommended Roadmap
The following directions can be a starting direction for your development. You are not just limited to these directives; feel free to create something that has not been mentioned in the following.

 <img src="/assets/fault-rr.png" width="50%">

Machine Learning (ML) is a way to build programs that learn patterns from data instead of being explicitly programmed with rules. In this challenge, you can use ML to analyze machine telemetry and predict failures before they happen.

The process can be broken into a few simple steps:

#### 1. Choose a Problem and Dataset

Start by defining what you want to predict. For example:

- Detect when a machine is about to fail
- Classify normal vs abnormal behavior

Then pick a dataset that supports your goal (e.g., the provided telemetry data or an open-source dataset).

#### 2. Research and Choose an ML Approach

Look into simple, well-documented techniques first. Some common starting points:

- Classification models (e.g., Logistic Regression, Decision Trees)
- Time-series methods (for sequential sensor data)

Focus on understanding _why_ a method fits your problem rather than choosing something complex - your project will be evaluated not just on results, but on the quality of your reasoning, the choices you make, and the insights you draw from your model.

#### 3. Clean and Prepare Data

Real-world data is often messy. You will likely need to:

- Handle missing or incorrect values
- Normalize or scale features (e.g., temperature, current)
- Select relevant features for your model

This step is critical - good data preparation often matters more than the model itself.

#### 4. Train the Model

Use your cleaned dataset to train the model:

- Split your data into training and testing sets
- Train on the training set
- Evaluate performance on unseen data

Python libraries like `scikit-learn` and `PyTorch` are commonly used here.

#### 5. Display and Interpret Results

Finally, present your results in a clear and meaningful way:

- Accuracy, precision/recall, or error metrics
- Graphs showing predictions vs actual behavior
- A simple dashboard or visualization

The goal is not just to build a model, but to show that it provides useful insight into machine behavior.

## Resources

- **Toyota Telemetry Data** (accessed via [Google Drive](https://drive.google.com/drive/folders/1WH95WIw2kX9aDbsBe2MpxEcpnStesaIB?usp=sharing))
  - The dataset contains time-series telemetry from an 8-DOF robotic manipulator, including joint states, end-effector pose, and actuator diagnostics at each timestamp. Specifically, it records joint positions, Cartesian position and orientation along with per-joint electrical/mechanical signals such as current, temperature, torque, and load percentage.
    <br>
- **Open Source Data**
  - Students are also permitted to make use of any Open Source Datasets. You can find datasets on websites like [Kaggle](https://www.kaggle.com/datasets), [AWS Open Data Registry](https://registry.opendata.aws/) or [Microsoft Research Open Data](https://msropendata.com/).
  - Some Example Datasets are listed below:
    1. [NASA Turbofan Jet Engine Dataset](https://www.kaggle.com/datasets/behrad3d/nasa-cmaps)
    2. [Azure Predictive Maintenance Dataset](https://www.kaggle.com/datasets/arnabbiswas1/microsoft-azure-predictive-maintenance)
    3. [Audio Dataset for Malfunctioning Machines](https://arxiv.org/abs/1909.09347)
    4. [Synthetic Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset)

    <br>

- **Test Setup for Motor Stall**
  - Students can also gain hands on experience with gathering failure data by using the provided Motor Stall Test Setup.
  - Further instructions can be found [here](/Fault_Prediction/MotorStallTestSetup/)

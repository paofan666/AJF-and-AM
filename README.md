# data
This study focuses on software defect prediction for Java projects.
The experimental data is sourced from the PROMISE public dataset, a well-known benchmark collection for software engineering research.

Five Java projects from the PROMISE dataset were selected for the experiments: Log4j, Lucene, Poi, Synapse, and Velocity.
- For within-project defect prediction, the lower version of each project is used as the training set, while the higher version is used as the test set.

  Example:
  If project X has versions 1.0 and 2.0 in the dataset,

  Version 1.0 → Training data

  Version 2.0 → Testing data

  This setup simulates a realistic scenario where defect prediction is performed for an upcoming release based on historical data from earlier releases.
- For cross-project software defect prediction, one of the above five items is taken as the training set, and the other four items are taken as the test set. 

# code
To evaluate the feasibility of the AST joint feature + Attention-BiLSTM software defect prediction algorithm proposed in this thesis, experiments were conducted under two prediction scenarios and three sets of comparative experiments were designed:

- Experimental comparison of AST combined features and individual features – The AST joint feature, the pure AST feature and the pure static measurement feature are taken as the data defect feature respectively, and the LSTM model is used to carry out the feature learning for the three features, and the software defect prediction experiment is completed.[At-Bi-LSTM-SA.py](code/At-Bi-LSTM-SA.py)

- Experimental comparison of defect models intra project – To compare the proposed model with baseline approaches in within-project settings.

- Experimental comparison of defect prediction cross project – To evaluate model generalization capability in cross-project settings.

# Evaluation Metrics

  The experiments adopt four commonly used evaluation metrics in software defect prediction:

  Accuracy – Overall correctness of predictions.

  Precision – Proportion of predicted defect-prone modules that are truly defective.

  Recall – Proportion of actual defect-prone modules correctly identified.

  F1-Score – Harmonic mean of Precision and Recall, providing a balanced measure of both.

  The F1-Score is particularly important as it comprehensively considers both Precision and Recall.Therefore, this thesis chooses F1 metric as the evaluation index of software defect prediction model.

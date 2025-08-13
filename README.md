# Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Adv20202/PrimeSpecPCR.git
   cd PrimeSpecPCR
   ```

3. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
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

- Experimental comparison of AST combined features and individual features – The AST joint feature [LSTM_SA.py](code/LSTM_SA.py), the pure AST feature [LSTM_AST.py](code/LSTM_AST.py) and the pure static measurement feature [LSTM_Static.py](code/LSTM_Static.py) are taken as the data defect feature respectively, and the LSTM model is used to carry out the feature learning for the three features, and the software defect prediction experiment is completed.

- Experimental comparison of defect models intra project – The AST joint feature is taken as the data defect feature, and Attention-CNN [At-CNN-SA.py](code/At-CNN-SA.py) and ABL models [At-Bi-LSTM-SA.py](code/At-Bi-LSTM-SA.py) are used to learn the defect feature respectively, and the experimental comparison of defect models within the project is completed..

- Experimental comparison of defect prediction cross project – In the cross-project software defect prediction experiment comparison, AST joint feature is used as data defect feature, Attention-CNN [TCA-Attention-CNN-SA.py](code/TCA-Attention-CNN-SA.py) and ABL [TCA-Attention-BiLSTM-SA.py](code/TCA-Attention-BiLSTM-SA.py) models are used to learn the defect feature, and cross-project software defect prediction is completed..

# Evaluation Metrics

  The experiments adopt four commonly used evaluation metrics in software defect prediction:

  Accuracy – Overall correctness of predictions.

  Precision – Proportion of predicted defect-prone modules that are truly defective.

  Recall – Proportion of actual defect-prone modules correctly identified.

  F1-Score – Harmonic mean of Precision and Recall, providing a balanced measure of both.

  The F1-Score is particularly important as it comprehensively considers both Precision and Recall.Therefore, this thesis chooses F1 metric as the evaluation index of software defect prediction model.

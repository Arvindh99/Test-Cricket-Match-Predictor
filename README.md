**🏏 Cricket Test Match Predictor**

A machine learning-powered web app that predicts whether a Test cricket match is likely to end in a draw or not, and if not, suggests which team has the upper hand. The app is built using Gradio and deployed on Hugging Face Spaces.


![image/png](https://cdn-uploads.huggingface.co/production/uploads/65e84c5037187ac7a13dca56/T_ZWot1Hdr2t1U94QtRRb.png)

***🚀  Features***

- Predicts if a Test match will end in a draw based on match configuration.
- If not a draw, predicts the winning team with the upper hand.
- Displays a probability breakdown chart.
- Automatically updates UI based on selected teams (e.g., Toss Winner dropdown).
- Interactive and easy-to-use UI powered by Gradio.

***🧠 Models Used***

The app uses machine learning classifiers trained on historical match data. Multiple models were evaluated using GridSearch and cross-validation to identify the best performers.

****Evaluated Models:****
- Logistic Regression
- Decision Tree
- Random Forest
- Naive Bayes
- K-Nearest Neighbors (KNN)
- Gradient Boosting

****Final Model Selection:****
- ✅ Draw Prediction:
K-Nearest Neighbors was found to be the best model based on evaluation metrics such as F1-score and ROC AUC. It provided the most balanced performance in distinguishing draw outcomes.

- ✅ Winner Prediction (when not a draw):
Gradient Boosting Classifier outperformed others with the highest accuracy and F1-score, making it the most reliable model for predicting the winning team when a result is expected.

****Input Features****

Venue, Team 1, Team 2, Toss Winner, Toss Decision

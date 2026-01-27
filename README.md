Mushroom Edibility Classification
-

The project includes an interactive Streamlit web application that provides:
-
- description of the objective of the project
- key observations and results
- a table showing test accuracies of all models
- interactive prediction demo of the best model

Technologies used:
-
- Python 3.13.7
- pandas 
- scikit-learn 
- Streamlit 
- ucimlrepo
- joblib


How to run the project:
-
1. Technologies mentioned before have to be installed.
2. Unzip the file `model_rf.zip`. There is a saved trained Random Forest model `model_rf.pkl` in the archive.
3. Run the app using `streamlit run app.py`.


Other files:
-
- log_reg.py

Script for training and evaluating the Logistic Regression model.

- rand_for.py

Script for training and evaluating the Random Forest model.

- label_prop.py

Script for training and evaluating the Random Forest model with Label Propagation.

- model_rf.zip with model_rf.pkl

Saved trained Random Forest model.

- dummy_columns.pkl

Stores the list of dummy feature columns used during training to ensure consistent input during prediction.

- cap_surface_map.pkl

Mapping dictionary for encoding the cap_surface feature.

- gill_attachment_map.pkl

Mapping dictionary for encoding the gill_attachment feature.

- ring_type_map.pkl

Mapping dictionary for encoding the ring_type feature.


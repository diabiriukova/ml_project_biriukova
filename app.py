import joblib
import streamlit as st
import pandas as pd

# Load trained Random Forest
model = joblib.load("model_rf.pkl")
dummy_columns = joblib.load("dummy_columns.pkl")

cap_surface_map = joblib.load("cap_surface_map.pkl")
gill_attachment_map = joblib.load("gill_attachment_map.pkl")
ring_type_map = joblib.load("ring_type_map.pkl")

st.set_page_config(
    page_title="Mushroom Classification",
    layout="centered"
)
st.title("Mushroom Edibility Classification")
st.subheader("Supervised vs Semi-Supervised Learning")

st.header("Project Objective")

st.markdown("""
The goal of this project was to evaluate whether **semi-supervised learning methods**
can achieve performance comparable to **supervised learning** on the mushroom dataset
under realistic labeling constraints.
""")

st.header("Dataset and Methods")

st.markdown("""
### Dataset
- Wagner D, Heider D, Hattab G. Secondary Mushroom [dataset]. 2021. UCI Machine Learning Repository. Available from: https://doi.org/10.24432/C5FP5Q.
- Categorical mushroom attributes
- Target attribute: edible or poisonous
- Dataset of 61068 labeled instances
- Missing values handled via **imputation** 
- Categorical features converted into dummy variables using **one-hot encoding**
- Testing subset = 30%

To compare semi-supervised learning with the supervised one, the dataset was splitted
to labeled and unlabeled. The labeled subset consisted of 100,200,500 and 1000 instances.

It is assumed that the amount of instances can be theoretically labeled by a human expert.

### Supervised Models
- Logistic Regression
- Random Forest 

### Semi-Supervised Methods
- Iterative pseudo-labeling with Logistic Regression
- Iterative pseudo-labeling with Random Forest + PCA
- Label Propagation combined with Random Forest + PCA
""")

st.header("Results")

st.markdown("""
Across all experiments, **semi-supervised methods consistently underperformed
compared to supervised learning trained on the same number of labeled samples.

Model performance was measured using test accuracy.
""")

data = {
    "Model": ["Logistic Regression on labeled  42746 instances ", "Logistic Regression on labeled 100 instances", "Iterative pseudo-labeling with Logistic Regression and 100 labeled instances(56 iterations)","Random Forest on labeled  42746 instances","Random Forest on labeled 100 instances", "Iterative pseudo-labeling with Random Forest and 100 labeled instances(0 iterations)", "Random Forest on labeled 200 instances", "Iterative pseudo-labeling with Random Forest and 200 labeled instances(27 iterations)","Label Propagation combined with Random Forest + PCA and 100 labeled instances","Label Propagation combined with Random Forest + PCA and 200 labeled instances","Label Propagation combined with Random Forest + PCA and 500 labeled instances","Label Propagation combined with Random Forest + PCA and 1000 labeled instances"],
    "Test Accuracy (%)": [76.9, 64.4,61.7, 99.9, 67.3, 67.3, 72.7,67.3, 52.3, 50.1,52.2,55.2]
}

df = pd.DataFrame(data)

st.subheader("Model Test Accuracies")
st.table(df)

st.markdown("""
Key observations:
- Pseudo-labeling and Label Propagation did not improve results for either Logistic Regression or Random Forest.
- Supervised Random Forest even with small amount of labeled insances achieved good results without requiring unlabeled data.
- Adding semi-supervised methods and unlabeled made results only worse.
- Despite the drastic reduction in labeled training data, the decrease in test accuracy was relatively small in Logistic Regression. 


This suggests that the model is able to capture the underlying structure of the data even with limited supervision, 
likely due to informative feature representations.
These results demonstrate that semi-supervised learning is not always beneficial
and strongly depends on dataset structure.
""")

st.header("Interactive Prediction Demo")

st.markdown("""
This demo allows you to test the **final supervised Random Forest model**. It was trained on labeeled dataset of 42746 instances and tested on 18321 instances.
The model produces accuracy of 99%.
""")

st.markdown("Enter mushroom characteristics to predict edibility.")

cap_diameter = st.number_input(
    "Cap diameter (cm)",
    min_value=0.1,
    max_value=50.0,
    step=0.1
)

stem_height = st.number_input(
    "Stem height (cm)",
    min_value=0.1,
    max_value=50.0,
    step=0.1
)

stem_width = st.number_input(
    "Stem width (mm)",
    min_value=0.1,
    max_value=50.0,
    step=0.1
)

cap_shape = st.selectbox("Cap Shape", ["bell","conical","convex","flat","sunken","spherical","others"])
cap_color = st.selectbox("Cap Color", ["brown","buff","gray","green","pink","purple","red", "white", "yellow", "blue","orange","black"])
cap_surface = st.selectbox("Cap Surface", ["fibrous","grooves","scaly", "smooth","shiny","leathery", "silky","sticky","wrinkled","fleshy"])
does_bruise_bleed = st.checkbox("Bruises or bleeding?", [True, False])
gill_attachment = st.selectbox("Gill Attachment", ["adnate", "adnexed","decurrent","free","sinuate","pores","none","unknown"])
gill_color= st.selectbox("Gill Color", ["brown","buff","gray","green","pink","purple","red", "white", "yellow", "blue","orange","black","none"])
stem_color = st.selectbox("Stem Color", ["brown","buff","gray","green","pink","purple","red", "white", "yellow", "blue","orange","black","none"])
has_ring = st.checkbox("Has Ring?", [True, False])
ring_type = st.selectbox("Ring Type", ["cobwebby","evanescent","flaring","grooved","large","pendant","sheathing","zone","scaly","movable","none","unknown"])
habitat = st.selectbox("Habitat",["glasses","leaves","meadows","paths","heaths","urban","waste","woods"])
season = st.selectbox("Season",["spring","summer","autumn","winter"])

input_data = {
    "cap-diameter": cap_diameter,
    "cap-shape": cap_shape,
    "cap-surface": cap_surface,
    "cap-color": cap_color,
    "does-bruise-bleed": does_bruise_bleed,
    "gill-attachment": gill_attachment,
    "gill-color": gill_color,
    "stem-height": stem_height,
    "stem-width": stem_width,
    "stem-color": stem_color,
    "has-ring": has_ring,
    "ring-type": ring_type,
    "habitat": habitat,
    "season": season
}

df_input = pd.DataFrame([input_data])

df_input["cap-surface"] = df_input["cap-surface"].map(cap_surface_map)
df_input["gill-attachment"] = df_input["gill-attachment"].map(gill_attachment_map)
df_input["ring-type"] = df_input["ring-type"].map(ring_type_map)

X_input_dummies = pd.get_dummies(df_input, dtype=int)

X_input_dummies = X_input_dummies.reindex(
    columns=dummy_columns,
    fill_value=0
)

pred = model.predict(X_input_dummies)[0]
prob = model.predict_proba(X_input_dummies)[0]

if st.button("Predict mushroom edibility"):
    pred = model.predict(X_input_dummies)[0]
    prob = model.predict_proba(X_input_dummies)[0]

    st.markdown("---")
    st.markdown("### Prediction result")

    if pred == 1:
        st.markdown(
            "<div style='background-color:red; color:white; padding:10px;'>Poisonous mushroom!</div>",
            unsafe_allow_html=True
        )
        st.write(f"Confidence: **{prob[1]*100:.2f}%**")
    else:
        st.markdown(
            "<div style='background-color:red; color:white; padding:10px;'>Edible mushroom!</div>",
            unsafe_allow_html=True
        )
        st.write(f"Confidence: **{prob[0]*100:.2f}%**")

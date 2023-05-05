import xgboost
import streamlit as st
import pandas as pd
import numpy as np
import sklearn
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt 
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score 
from sklearn.metrics import accuracy_score
from xgboost import XGBRFClassifier
from xgboost import XGBClassifier
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold



def main():
    st.title("Breast Cancer Classification Web App")
    st.sidebar.title("Breast Cancer Classification Web App")
    st.markdown("Are you have breast cancer?")
    st.sidebar.markdown("Are you have breast cancer?")


    def load_data():
        df = pd.read_csv('dataR2.csv')
        return data
        
    def split(df):
        y = df.Classification
        x = df.drop(columns =['Classification'])
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)
        return x_train, x_test, y_train, y_test








    def plot_metrics(metrics_list):
        st.set_option('deprecation.showPyplotGlobalUse', False)



        if 'Confusion Matrix' in metrics_list:
            st.subheader("Confusion Matrix") 
            ConfusionMatrixDisplay(model, x_test, y_test, display_labels=class_names)
            st.pyplot()
        
        if 'ROC Curve' in metrics_list:
            st.subheader("ROC Curve") 
            plot_roc_curve(model, x_test, y_test)
            st.pyplot()

        if 'Precision-Recall Curve' in metrics_list:
            st.subheader("Precision-Recall Curve")
            plot_precision_recall_curve(model, x_test, y_test)
            st.pyplot()

        if 'Seleksi Fitur' in metrics_list:
            st.subheader("Seleksi Fitur")
            plt.figure(figsize=(16, 14))
            plt.barh(y=dset['attr'], width=dset['importance'], color='#1976D2')
            plt.title('RFECV - Feature Importances', fontsize=20, fontweight='bold', pad=20)
            plt.xlabel('Importance', fontsize=14, labelpad=20)
            st.pyplot(plt.gcf())
            


    df = load_data()
    class_names = ['Health Control', 'Patient']

    x_train, x_test, y_train, y_test = split(df)


        
    
    st.sidebar.subheader("Choose Classifier")
    classifier = st.sidebar.selectbox("Classifier", ("Recursive Feature Elimination Cross Validation", "Random Forest", "Random Forest dengan XGBoost"), key="3")

    if classifier == 'Recursive Feature Elimination Cross Validation':
        st.sidebar.subheader("Model Hyperparameters")
        

        if st.sidebar.button("Classfiy", key='classify'):
            st.subheader("Recursive Feature Elimination Cross Validation")
            x_full = df.drop(columns =['Classification'])
            y_full = df.Classification
            rfc = RandomForestClassifier(random_state=0)
            rfecv = RFECV(estimator=rfc, step=1, cv=StratifiedKFold(10), scoring='accuracy')
            rfecv.fit(x_full, y_full)
            x_full.drop(x_full.columns[np.where(rfecv.support_ == False)[0]], axis=1, inplace=True)
            dset = pd.DataFrame()
            dset['attr'] = x_full.columns
            dset['importance'] = rfecv.estimator_.feature_importances_
            dset = dset.sort_values(by='importance', ascending=False)
            plot_metrics('Seleksi Fitur')



    if classifier == 'Random Forest':
        st.sidebar.subheader("Model Hyperparameters")
        n_estimators  = st.sidebar.number_input("The number of trees in the forest", 100, 5000, step=10, key='n_estimators')
        max_depth = st.sidebar.number_input("The maximum depth of the tree", 1, 20, step=1, key='max_depth')


        if st.sidebar.button("Classfiy", key='classify'):
            st.subheader("Random Forest")
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, n_jobs=-1, random_state=3)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy ", accuracy.round(7))
            st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(7))
            st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(7))

    if classifier == 'Random Forest dengan XGBoost':
        st.sidebar.subheader("Model Hyperparameters")
        num_parallel_tree = st.sidebar.number_input("The number of trees in the forest", 100, 200, step=10, key='num_parallel_tree')
        max_depth = st.sidebar.number_input("The maximum depth of the tree", 1, 20, step=1, key='max_depth')

        
        if st.sidebar.button("Classfiy", key='classify'):
            st.subheader("Random Forest dengan XGBoost")
            model = XGBClassifier(num_parallel_tree=num_parallel_tree, max_depth=max_depth, n_jobs=-1)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy ", accuracy.round(7))
            st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(7))
            st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(7))

    if st.sidebar.checkbox("Pilih Data", False, key='lihat2'):
        st.subheader("Pilih data coimbra data set (Classification)")
        st.write(df)
        with st.form('form'):
          sel_column = st.multiselect('Select column', df.columns,
          help='Select a column to form a new dataframe. Press submit when done.')
          drop_na = st.checkbox('Drop rows with missing value', value=True)
          submitted = st.form_submit_button("Submit")
  


    if st.sidebar.checkbox("Show raw data Sebelum", False, key='lihat1'):
        st.subheader("coimbra data set (Classification)")
        st.write(df)

    
    if st.sidebar.checkbox("Lihat data baru", False, key='lihat3'):
        st.subheader("coimbra data set (Classification)")
        st.write(dfnew)

    if st.sidebar.checkbox("AKurasi Random Forest", False, key='lihat78'):
        st.subheader("Random Forest")
        model2 = RandomForestClassifier(n_estimators=10, max_depth=10,random_state=3)
        model2.fit(x_train2, y_train2)
        accuracy = model2.score(x_test2, y_test2)
        y_pred2 = model2.predict(x_test2)
        st.write("Accuracy ", accuracy.round(7))
        st.write("Precision: ", precision_score(y_test2, y_pred2, labels=class_names).round(7))
        st.write("Recall: ", recall_score(y_test2, y_pred2, labels=class_names).round(7))
        st.write(dfnew)

    if st.sidebar.checkbox("Hasil baru", False, key='lihat80'):
        st.subheader("Random Forest")
        model2 = RandomForestClassifier(n_estimators=10, max_depth=10,random_state=3)
        model2.fit(x_train2, y_train2)
        Age = st.number_input("Masukan Umur") 
        BMI = st.number_input("Masukan BMI") 
        Restitin = st.number_input("Restitin") 
        Glucose = st.number_input("Glucose")
        hasil_1 = model2.predict(Age, BMI, Restitin, Glucose)
        st.write(hasil_1)




if __name__ == '__main__':
    main()

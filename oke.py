import streamlit as st
import altair
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
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold

def main():
    st.title("Breast Cancer Classification Web App")
    st.sidebar.title("Breast Cancer Classification Web App")
    st.markdown("Prediksi Kanker Payudara dengan Seleksi Fitur RFECV dan Random Forest dengan XGBoost")
    st.sidebar.markdown("Prediksi Kanker Payudara dengan Seleksi Fitur RFECV dan Random Forest dengan XGBoost")
    
    df = pd.read_csv('dataR2.csv')
    dfnew = pd.read_csv('coimbra dataset breast sudah rfe.csv')


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
            
        
        
    st.sidebar.subheader("Sebelum RFECV")

    if st.sidebar.checkbox("Lihat Data Sebelum RFECV", False, key='lihatlama1'):
        st.subheader("Coimbra Breast Cancer Dataset")
        st.write(df)
    
    if st.sidebar.checkbox("RFECV (Recursive Feature Elimination Cross Validation)", False, key='lihatlama2'):
         st.subheader("Seleksi Fitur RFECV")
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


    if st.sidebar.checkbox("Random Forest", False, key='lihatlama3'):
            st.subheader("Random Forest Sebelum RFECV")
            y = df.Classification
            x = df.drop(columns =['Classification'])
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)
            cv = KFold(n_splits=10, random_state=1, shuffle=True)
            model = RandomForestClassifier(n_estimators=100, max_depth=1, random_state=3)
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
            st.write('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
            st.write("Accuracy ", accuracy_score(y_test, y_pred).round(7))
            st.write("Precision: ", precision_score(y_test, y_pred).round(7))
            st.write("Recall: ", recall_score(y_test, y_pred).round(7))

    if st.sidebar.checkbox("Random Forest dengan XGBoost", False, key='lihatlama4'):
            st.subheader("Random Forest dengan XGBoost Sebelum RFECV")
            y = df.Classification
            x = df.drop(columns =['Classification'])
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)
            model = XGBClassifier(num_parallel_tree=100, max_depth=1,random_state=3 )
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            st.write("Accuracy ", accuracy_score(y_test, y_pred).round(7))
            st.write("Precision: ", precision_score(y_test, y_pred).round(7))
            st.write("Recall: ", recall_score(y_test, y_pred).round(7))

    st.sidebar.subheader("Data Baru Setelah RFECV")

    if st.sidebar.checkbox("Lihat Data Baru Setelah RFECV", False, key='lihatbaru1'):
        st.subheader("Coimbra Breast Cancer Dataset Setelah RFECV")
        st.write(dfnew)

    if st.sidebar.checkbox("Random Forest", False, key='lihatbaru2'):
        st.subheader("Random Forest Setelah RFECV")
        y2 = dfnew.Classification
        x2 = dfnew.drop(columns =['Classification'])
        x_train2, x_test2, y_train2, y_test2 = train_test_split(x2, y2, test_size=0.3, random_state=1)
        modele = RandomForestClassifier(n_estimators=100, max_depth=1,random_state=3)
        modele.fit(x_train2, y_train2)
        y_pred2 = modele.predict(x_test2)
        st.write("Accuracy ", accuracy_score(y_test2, y_pred2).round(7))
        st.write("Precision: ", precision_score(y_test2, y_pred2).round(7))
        st.write("Recall: ", recall_score(y_test2, y_pred2).round(7))

    if st.sidebar.checkbox("Random Forest dengan XGBoost", False, key='lihatbaru3'):
        st.subheader("Random Forest dengan XGBoost Setelah RFECV")
        y2 = dfnew.Classification
        x2 = dfnew.drop(columns =['Classification'])
        x_train2, x_test2, y_train2, y_test2 = train_test_split(x2, y2, test_size=0.3, random_state=1)
        modele = XGBClassifier(num_parallel_tree=100, max_depth=1,random_state=3)
        modele.fit(x_train2, y_train2)
        y_pred2 = modele.predict(x_test2)
        st.write("Accuracy ", accuracy_score(y_test2, y_pred2).round(7))
        st.write("Precision: ", precision_score(y_test2, y_pred2).round(7))
        st.write("Recall: ", recall_score(y_test2, y_pred2).round(7))
        
    st.sidebar.subheader("Prediksi Kanker Payudara")


    if st.sidebar.checkbox("Prediksi Kanker Payudara", False, key='prediksibaru1'):
        st.header("Prediksi Kanker Payudara")
        st.write("Masukan Data Untuk Prediksi")
        y1 = dfnew.Classification
        x1 = dfnew.drop(columns =['Classification'])
        x1_train,x1_test,y1_train,y1_test=train_test_split(x1,y1,test_size=0.3,random_state=1)
        model1=XGBClassifier(num_parallel_tree=100, max_depth=1,random_state=3)
        model1.fit(x1_train,y1_train)
        dfst1 = pd.DataFrame(columns=["Age", "BMI", "Glucose", "Resistin"])
        Age=st.number_input("Masukan Umur") 
        BMI=st.number_input("Masukan BMI",format="%.7f") 
        Glucose=st.number_input("Glucose") 
        Resistin=st.number_input("Resistin",format="%.5f")
        dfst2 = pd.DataFrame(data=[[Age, BMI, Glucose, Resistin]],columns=["Age", "BMI", "Glucose", "Resistin"])
        dfoke = pd.concat([dfst1,dfst2], axis=0)

        prediction1=model1.predict(dfoke)

        if st.button("Predict"):
          if prediction1==1:
             st.write("Healthy controls/1")
          elif prediction1==2:
             st.write("Patients/2")
    

if __name__ == '__main__':
    main()

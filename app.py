import streamlit as st 
from PIL import Image 
import pandas as pd 
import pickle 
from sklearn.impute import SimpleImputer 
model=pickle.load(open('svm_model.pkl','rb')) 
back_ground=Image.open("Cyberattack.jpeg") 
def main(): 
    st.title("Detection of False Data Injection") 
    st.image(back_ground,use_column_width=True) 
    # Upload CSV file 
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"]) 

    if uploaded_file is not None: 
        # Read CSV file 
        df = pd.read_csv(uploaded_file) 
        features = ['kwhTotal', 'dollars', 'chargeTimeHrs', 'distance', 'Mon', 'Tues', 'Wed', 'Thurs', 'Fri', 'Sat', 'Sun', 'managerVehicle'] 
        new_x=df[features] 
        x_session=df[['sessionId']] 
        if new_x.isnull().values.any(): 
            # Impute NaN values 
            imputer = SimpleImputer(strategy='mean') # You can choose a different strategy as needed 
            new_x_imputed = imputer.fit_transform(new_x) 

            # Predict using the imputed data 
            y_pred = model.predict(new_x_imputed) 
        else: 
        # Predict directly if there are no NaN values 
            y_pred = model.predict(new_x) 
        # Display the dataframe 
        result= x_session.assign(predict=y_pred)
        false_detect=result.loc[result['predict']==1] 
        print(len(false_detect)) 
        print(false_detect) 
        st.write(false_detect) 
        st.write(len(false_detect)) 
if __name__ == "__main__": 
    main()



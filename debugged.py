import streamlit as st
import joblib,os 

prediction = ""

# Vectorizer
complain_vectorizer = open("vectorizer.pkl", "rb")
complain_cv = joblib.load(complain_vectorizer)


#load our model
def load_prediction_model(model_file):
    loaded_model = joblib.load (model_file)#(open(os.path.join(model_file), "rb"))
    return loaded_model

def main():
    st.title("Complain Classifier ML App")
    st.subheader("ML APP with Streamlit")


    sidebar= ["Home", "Database"]

    choice= st.sidebar.selectbox("Choose Activity", sidebar)


    if choice == "Home":
        st.info("Prediction with ML")

        complain_text = st.text_area("Enter Complain", "Type Here")
        # prediction_labels = {"Closed with explanation","Closed with non-monetary relief",
        # "Closed with monetary relief", "Closed without relief",
        # "Closed","Closed with relief", "In progress", "Untimely response"}

        if st.button("Predict"):
            
            st.text("Orignal text :\n{}".format(complain_text))
            vect_text = complain_cv.transform([complain_text]).toarray()
            predictor = load_prediction_model("text.joblib")
            prediction = predictor.predict(vect_text)


    # if choice == "Database":
    #     st.info("Database Management")
    

# st.write(prediction)

if __name__ == '__main__':
    main()



 
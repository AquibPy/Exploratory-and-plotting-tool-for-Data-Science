import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.use("Agg")
from PIL import Image
import time
import webbrowser
from sklearn.cluster import KMeans

def app():
    st.title('Exploratory Data Analysis Tool')
    image = Image.open('logo.jpeg')
    st.sidebar.image(image,use_column_width=True)
    activities = ["Exploratory Data Analysis","Plots","Clusters","About us"]
    choice = st.sidebar.selectbox("Select Activities",activities)
    if choice == 'Exploratory Data Analysis':
            data = st.file_uploader('Upload a Dataset',type=['csv','txt'])
            if data is not None:
                df = pd.read_csv(data)
                st.dataframe(df.head())

                if st.checkbox('Show Shape'):
                    st.write(df.shape)
                if st.checkbox('Show Columns'):
                    all_columns = df.columns.to_list()
                    st.write(all_columns)
                if st.checkbox("Show Missing Values"):
                    all_columns = df.columns.to_list()
                    df[all_columns]=df[all_columns].replace(0,np.nan)
                    st.write(df.isna().sum())
                    st.subheader('To Handle Missing Value use Imputer')
                    st.write('For Example:')
                    st.code('''
                    from sklearn.impute import SimpleImputer
                    imputer = SimpleImputer(missing_values =np.NaN,strategy='median')
                    imputer.fit(df)
                    x = imputer.transform(df)
                    df = pd.DataFrame(x,columns=df.columns)
                    ''',language='python')
                if st.checkbox('Summary'):
                    latest_iteration = st.empty()
                    bar = st.progress(0)

                    for i in range(100):
                    # Update the progress bar with each iteration.
                        latest_iteration.text(f'Loading... {i+1}')
                        bar.progress(i + 1)
                        time.sleep(0.1)
                    st.write(df.describe().transpose())
                if st.checkbox("Show Selected Columns"):
                    sel_column =st.multiselect("Select Columns",all_columns)
                    df_new = df[sel_column]
                    st.dataframe(df_new)
                if st.checkbox('Correlation Matrix'):
                    st.write(df.corr())
                if st.checkbox('Correlation Plot(Using Matplotlib)'):
                    plt.matshow(df.corr())
                    st.pyplot()
                if st.checkbox('Correlation Plot(Using Seaborn)'):
                    sns.heatmap(df.corr(),annot=True)
                    st.pyplot()
                if st.checkbox('Pie Plot'):
                    all_columns = df.columns.to_list()
                    col_to_plot = st.selectbox('Select 1 Column',all_columns)
                    pie_plot = df[col_to_plot].value_counts().plot.pie(autopct="%1.1f%%")
                    st.write('Pie Plot')
                    st.pyplot()
            
            st.subheader('If You like this Tool Lets Celebrate!!')        
            btn = st.button("Celebrate!")
            if btn:
                st.balloons()
    
    elif choice == 'Plots':
        st.header('Data Visualization')
        data = st.file_uploader("Upload a Dataset", type=['csv','txt','xlsx'])
        if data is not None:
            df = pd.read_csv(data)
            st.dataframe(df.head())

            if st.checkbox("Show Value Counts"):
                st.write(df.iloc[:,-1].value_counts().plot(kind = 'bar'))
                st.pyplot()

            all_columns = df.columns.to_list()
            type_plot = st.selectbox("Select type of Plot",["Area",'Bar','Line','hist','box','kde'])
            select_column = st.multiselect("Select Columns To Plot",all_columns)
            if st.button('Generate Plot'):
                st.success("Generating Plot of {} for {}".format(type_plot,select_column))

                if type_plot == "Area":
                    cdata = df[select_column]
                    st.area_chart(cdata)
                
                elif type_plot == "Bar":
                    cdata = df[select_column]
                    st.bar_chart(cdata)
                
                elif type_plot == "Line":
                    cdata = df[select_column]
                    st.line_chart(cdata)
                else:
                    cplot = df[select_column].plot(kind = type_plot)
                    st.write(cplot)
                    st.pyplot()
    elif choice == "Clusters":
        data = st.file_uploader("Upload a Dataset", type=['csv'])
        if data is not None:
            df = pd.read_csv(data)
            st.dataframe(df.head())
            X = df.iloc[:,[3,4]].values
            st.subheader('Note:')
            st.write('Change X(Independent Variable) of our own from source code if you add another csv except Mall_Customers.csv and find number of clusters using Elbow Method.I have Provided github link in about us section to download the source code.')
            if st.button("Elbow Method"):
                wcss = []
                for i in range(1,11):
                    kmeans = KMeans(n_clusters= i, init='k-means++',max_iter=300,n_init=10,random_state=0)
                    kmeans.fit(X)
                    wcss.append(kmeans.inertia_)
                plt.plot(range(1,11),wcss)
                plt.title('The Elbow Method')
                plt.xlabel('Number of Clusters')
                plt.ylabel('WCSS')
                st.pyplot()

            kmeans = KMeans(n_clusters=5,init='k-means++',max_iter=300,n_init=10,random_state=0)
            y_kmeans = kmeans.fit_predict(X)
            if st.button('Clusters'):
                 plt.scatter(X[y_kmeans == 0, 0],X[y_kmeans == 0, 1],s = 100, c ='red',label="Cluster 1")
                 plt.scatter(X[y_kmeans == 1, 0],X[y_kmeans == 1, 1],s = 100, c ='blue',label="Cluster 2")
                 plt.scatter(X[y_kmeans == 2, 0],X[y_kmeans == 2, 1],s = 100, c ='green',label="Cluster 3")
                 plt.scatter(X[y_kmeans == 3, 0],X[y_kmeans == 3, 1],s = 100, c ='cyan',label="Cluster 4")
                 plt.scatter(X[y_kmeans == 4, 0],X[y_kmeans == 4, 1],s = 100, c ='magenta',label="Cluster 5")
                 plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s = 300, c = 'yellow',label="Centroid")
                 plt.title('Clusters of clients')
                 plt.xlabel('Annual Income(k$)')
                 plt.ylabel('Spending Score(1-100)')
                 plt.legend()
                 st.pyplot()

    else:
        st.header("SCRIPTHON")

        st.write("""SCRIPTHON is a coding community founded by students of MCA, Aligarh Muslim University.
        We are a group of talented and enthusiastic students who have experienced in a certain area like ML/AI, Web Development, Android Development, Ethical Hacking & many more.
        All are willing to help each other to grow simultaneously and share their knowledge because Richard Feynman said "THE ULTIMATE TEST OF YOUR KNOWLEDGE IS YOUR CAPACITY TO CONVEY IT TO ANOTHER". 
        We've made many interesting projects which you can see on our website and currently we've community of five members, if you want to join our community then let us know.""")

        st.subheader('Our Team Members')
        st.subheader("1. Mohd Aquib")
        st.subheader("2. Nikhil Upadhyay")
        st.subheader("3. Mahiya Khan")
        st.subheader("4. Mohd Maaz Azhar")
        st.subheader("5. Dilanshi Varshney")



        st.subheader("Made with ❤️ by Mohd Aquib.")

        github = "https://github.com/AquibPy/Exploratory-and-plotting-tool-for-Data-Science"
        linkedIn = "https://www.linkedin.com/in/aquib-mohd-182b2a71/"

        if st.button('Github'):
            webbrowser.open_new_tab(github)
        if st.button('LinkedIn'):
            webbrowser.open_new_tab(linkedIn)

        
        





if __name__ == '__main__':
	app()
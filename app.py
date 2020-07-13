import streamlit as st
import glob
import pickle
from bokeh.plotting import figure, output_file, show, output_notebook,curdoc
from bokeh.models import ColumnDataSource, Plot, LinearAxis, Grid, HoverTool, Legend
from bokeh.models.glyphs import MultiLine
from bokeh.palettes import Category10
from bokeh.models import Range1d
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import altair as alt
from sklearn.manifold import TSNE
import plotly.express as px
from PIL import Image



def main():

    files = []
    for file in glob.glob("models/histories_F/*pkl"):
        files.append(file.rsplit('/')[2])
    print(files)
    page = st.sidebar.selectbox("App Selections", ["Homepage", "EDA", "Model Performance", "Embeddings", "Similarity", "Recommendations"])
    if page == "Model Performance":
        st.header("Exploring various models")
        visualize_data(files)
    elif page == "EDA":
        st.header("Exploratory Data Analysis")
        eda()
    elif page == 'Embeddings':
        st.header('Embeddings Visualisation')
        embed()
    elif page == 'Similarity':
        st.header('Behavioral Data')
        st.subheader('Recommendation Systems rely on collaborative filtering')
        similarity_filter()
    elif page == "Homepage":
        homepage()
    else:
        st.header("Recommendations for users")
        st.subheader('These recommendations are generated using a Deep Neural Net')
        recommendations()



@st.cache
def load_data():
    df = pd.read_csv("eda.csv", nrows=1000)
    return df

def recommendations():
    #st.write(df_rec)
    user_to_select = [1,2,3,4]
    desired_label = st.selectbox("Which user's similarity you want to see", [1,2,3,4])
    #sel_user = st.sidebar.selectbox("Which user's similarity you want to see", user_to_select)
    df_rec = pd.read_csv('recommendations_prod.csv')
    df_f = df_rec[df_rec['userID']==desired_label]
    #st.write(df_rec[df_rec['userID']==sel_user])
    #"""The user has rated""", df_f['Number of Films Watched'][0], """products"""
    st.write("Products already rated by the user are:")
    idx = desired_label - 1
    lst = df_f['Films Watched'][idx].split(',')
    #df_lst = pd.DataFrame(lst,columns = ['Products'])
    st.write(lst)
    st.write("Recommended products for the user are:")
    dic_r = df_f['Recommendations'][idx].split(',')
    st.write(dic_r)

def homepage():
    image = Image.open('logo6.png')
    st.image(image)
    st.title(' An _Advanced_ Recommender Engine')



def cosine_similarities(item_id, embeddings):
    EPSILON = 1e-07
    """Similarities between users based on their buying behavior and metadata"""
    #query_vector = em[item_id]
    #dot_products = item_embeddings @ query_vector
    print(item_id)
    #print(embeddings[item_id])
    query_vector = embeddings[str(item_id)]
    print(query_vector.shape)
    print(embeddings.shape)
    dot_products = embeddings.T @ query_vector

    query_vector_norm = np.linalg.norm(query_vector)
    all_item_norms = np.linalg.norm(embeddings, axis=0)
    norm_products = query_vector_norm * all_item_norms
    return dot_products / (norm_products + EPSILON)

def similarity_filter():
    #from keras.models import load_model
    #load_path = "models/MF_F/"
    #model = load_model(load_path+'MF_128_07_11_15_05.h5')
    #weights = model.get_weights()
    #user_embeddings = weights[0]
    #item_embeddings = weights[1]
    u_e = pd.read_csv('user_embeddings.csv')
    u_e = u_e.drop(['Unnamed: 0'], axis = 1)
    i_e = pd.read_csv('item_embeddings.csv')
    i_e = i_e.drop(['Unnamed: 0'], axis = 1)
    #st.write(u_e)
    desired_label = st.selectbox('Select Similarity Space:', ['Select', 'User Similarity', 'Product Similarity'])
    if desired_label == 'Select':
        print('do nothing')
    elif desired_label =='User Similarity':
        user_to_select = [1,2,3,4,5,6,7,8,9,10]
        sel_user = st.sidebar.selectbox("Which user's similarity you want to see", user_to_select)

        sims = cosine_similarities(sel_user, u_e)
        sorted_indexes = np.argsort(sims)[::-1]
        idxs = sorted_indexes[0:21]
        df = pd.DataFrame(list(zip(idxs, sims[idxs])),
               columns =['UserID', 'Similarity Index'])
        st.write(df)
        #st.dataframe(df.style.highlight_max(axis=0))
    elif desired_label == 'Product Similarity':
        product_to_select = [50,258,100,181,1,300,121,174,8,66,252]
        sel_prod = st.sidebar.selectbox("Which product's similarity you want to see", product_to_select)
        sims = cosine_similarities(sel_prod, i_e)
        sorted_indexes = np.argsort(sims)[::-1]
        idxs = sorted_indexes[0:21]
        indexed_items = pd.read_csv("idx_app_items.csv")
        titles = indexed_items['product']
        df2 = pd.DataFrame(list(zip(idxs, titles[idxs], sims[idxs])),
               columns =['ProductId', 'Main Product Cohort', 'Similarity Index'])
        st.write(df2)





def embed():
    from keras.models import load_model
    load_path = "models/MF_F/"
    model = load_model(load_path+'MF_128_07_11_15_05.h5')
    weights = model.get_weights()
    user_embeddings = weights[0]
    item_embeddings = weights[1]
    desired_label = st.selectbox('Visualise Embedding Space:', ['Select Space', 'Products', 'Users'])
    if desired_label == 'Select Space':
        print('do nothing')
    elif desired_label == 'Products':
        item_tsne = TSNE(perplexity=30).fit_transform(item_embeddings)
        tsne_df = pd.DataFrame(item_tsne, columns=["tsne_1", "tsne_2"])
        tsne_df["item_id"] = np.arange(item_tsne.shape[0])
        all_ratings = pd.read_csv("all_ratings copy.csv")
        tsne_df = tsne_df.merge(all_ratings.reset_index())

        space = px.scatter(tsne_df, x="tsne_1", y="tsne_2",
                   color="rating",
                   hover_data=["item_id",
                               "video_release_date", "popularity"])
        st.plotly_chart(space)
    elif desired_label == 'Users':
        user_tsne = TSNE(perplexity=30).fit_transform(user_embeddings)
        user_tsne_df = pd.DataFrame(user_tsne, columns=["tsne_1", "tsne_2"])
        user_tsne_df["user_id"] = np.arange(user_tsne.shape[0])
        all_ratings = pd.read_csv("all_ratings copy.csv")
        user_tsne_df = user_tsne_df.merge(all_ratings.reset_index())
        space = px.scatter(user_tsne_df, x="tsne_1", y="tsne_2",
                   color="popularity",
                   hover_data=["user_id", "video_release_date",
                               "item_id", "popularity"])

        st.plotly_chart(space)



def eda():
    data = load_data()
    if st.checkbox('Show Raw Data'):

        st.subheader('Raw data')
        #st.write(data)
        st.dataframe(data.style.highlight_max(axis=0))
    #st.subheader('Raw Data')
    #st.write(data)

    st.subheader('Distribution of ratings')
    hist_values = np.histogram(data['rating'], bins=6, range=(1,6))[0]
    st.bar_chart(hist_values)
    #print(data['rating'].describe())
    #plt.hist(data['rating'], bins=5)
    #st.pyplot()
    #st.bar_chart(np.histogram(data['rating']))

    #rating_to_filter = st.slider('rating', 0, 5, 2)
    #filtered_data = data[data['rating'] <= rating_to_filter]
    #st.subheader('Most popular products' % rating_to_filter)
    #st.map(filtered_data)
    st.subheader('Popularity')
    #x = st.slider(data['rating'])
    desired_label = st.selectbox('Filter to:', ['Users', 'Products'])
    if desired_label == 'Users':
        pop_users = pd.read_csv('pop_users.csv')
        pop_users = pop_users[0:35]
        #st.write(pop_users)
        bars = alt.Chart(pop_users).mark_bar(color='#4db6ac').encode(
            x=alt.X('Count:Q', axis=alt.Axis(title='Total Products Rated')),
            y=alt.Y('userID:O',
                    sort=alt.EncodingSortField(
                        field="Users",  # The field to use for the sort
                        order="descending"  # The order to sort in
                        )
                    )
        ).properties(height=800, width = 600)
        st.write(bars)
    else:
        pop_prods = pd.read_csv('pop_prods.csv')
        pop_prods = pop_prods[0:35]
        #st.write(pop_users)
        bars = alt.Chart(pop_prods).mark_bar(color='#4db6ac').encode(
            x=alt.X('Count:Q', axis=alt.Axis(title='# Users rated the products high')),
            y=alt.Y('product:O',
                    sort=alt.EncodingSortField(
                        field="Products",  # The field to use for the sort
                        order="descending"  # The order to sort in
                        )
                    )
        ).properties(height=800, width = 600)
        st.write(bars)


def visualize_data(files):

    models_to_select = []
    for file in files:
        models_to_select.append(file)

    models = st.sidebar.multiselect("Which models do you want to select", models_to_select)
    #y_lim = st.sidebar.selectbox("Choose your y_limit: ", np.arange(0, 10, 0.5))
    y_lim = st.sidebar.slider("Choose your y_limit: ", min_value = 0, max_value = 4, value = 2, step = 1)

    xs = []
    ys = []

    x_p = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
    for val in models:
        with open('models/histories_F/' + val , 'rb') as f:
            load_f = pickle.load(f)
            ys.append(load_f['loss'])
            xs.append(x_p)
            ys.append(load_f['val_loss'])
            xs.append(x_p)

    p = figure(plot_width=750, plot_height=800, title = "Progressive loss for various models")
    p.xaxis.axis_label = "Iterations"
    p.yaxis.axis_label = "Loss"

    label_list = []
    for file in models:
        label_list.append('loss_'+ file)
        label_list.append('val_loss_' + file)

    data = {'xs': xs,
            'ys': ys,
            'labels': label_list,
            'colors': Category10[2* len(files)]
            }

    source = ColumnDataSource(data)
    p.multi_line(xs='xs', ys='ys', legend_field = 'labels', color = 'colors', source=source, line_width = 4)
    p.y_range = Range1d(0, y_lim)
    #st.bokeh_chart(p)
    st.write(p)

if __name__ == "__main__":
    main()

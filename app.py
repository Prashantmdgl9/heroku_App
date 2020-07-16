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
    df = pd.read_csv("eda.csv")
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
    st.write("Recommended products for the user are:")
    df_rec_only = pd.read_csv('recommendations_only.csv')
    df_x = df_rec_only[df_rec_only['userID']==desired_label]
    #dic_r = df_f['Recommendations'][idx].split(',')
    chart = alt.Chart(df_x).mark_point(opacity=0.8,
    strokeWidth=20).encode(
    x=alt.X('Rating:Q'),
    #y='Recommendation:O',
    y=alt.Y('Recommendation:O', axis=alt.Axis(title='Recommendations'),
            sort=alt.EncodingSortField(
                field="Rating",  # The field to use for the sort
                order="descending"  # The order to sort in
                )
            ),
    color='Recommendation'
    ).interactive().properties(height = 400, width = 600)

    st.write(chart)
    st.write("Products already rated by the user are:")
    idx = desired_label - 1
    lst = df_f['Films Watched'][idx].split(',')
    #df_lst = pd.DataFrame(lst,columns = ['Products'])
    st.write(lst)

def homepage():
    image = Image.open('logo2.png')
    st.image(image)
    image2 = Image.open('shot1.png')
    st.image(image2)
    st.title('&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;An _Advanced_ Recommender Engine')
    #st.markdown('&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;An _Advanced_ Recommender Engine')




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

        interval = alt.selection_interval(encodings=['x', 'y'])
        chart = alt.Chart(df).mark_point().encode(
                     x=alt.X('UserID'),
                     y=alt.Y('Similarity Index'),
                     color=alt.condition(interval, 'count()', alt.value('lightgray')),
                     tooltip='UserID'
                ).interactive().properties(height = 500, width = 600, selection = interval)
        st.markdown('_Please select an area of the scatter plot_')
        hist = alt.Chart(df).mark_bar().encode(
                   x=alt.X('Similarity Index:Q', axis = alt.Axis(title='Similarity Index')),
                   y=alt.Y('UserID:O'),
                   color='Similarity Index'
                ).transform_filter(
                interval
                )
        st.write(alt.vconcat(chart,hist))
        #st.write(hist)
        if st.checkbox('Show tabular form'):
            st.subheader('Similarity with other users')
            #st.write(data)
            st.dataframe(df)
        #st.write(df)


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
        interval = alt.selection_interval(encodings=['x', 'y'])
        chart = alt.Chart(df2).mark_point().encode(
                     x=alt.X('ProductId'),
                     y=alt.Y('Similarity Index'),
                     color=alt.condition(interval, 'count()', alt.value('lightgray')),
                     tooltip='ProductId'
                ).interactive().properties(height = 500, width = 600, selection = interval)
        st.markdown('_Please select an area of the scatter plot_')
        hist = alt.Chart(df2).mark_bar().encode(
                   x=alt.X('Similarity Index:Q', axis = alt.Axis(title='Similarity Index')),
                   y=alt.Y('ProductId:O'),
                   color='Similarity Index'
                ).transform_filter(
                interval
                )
        st.write(alt.vconcat(chart,hist))
        #st.write(hist)
        if st.checkbox('Show tabular form'):
            st.subheader('Similarity with other users')
            #st.write(data)
            st.dataframe(df2)

        #st.write(alt.Chart(df2).mark_bar().encode(
        #             x=alt.X('ProductId', bin=True),
        #             y=alt.Y('Similarity Index', bin = True),
        #             color = 'count()'
        #        ).interactive().properties(height = 400, width = 600))
        #st.write(df2)





def embed():
    i_e = pd.read_csv('item_embeddings.csv')
    i_e = i_e.drop(['Unnamed: 0'], axis = 1)
    u_e = pd.read_csv('user_embeddings.csv')
    u_e = u_e.drop(['Unnamed: 0'], axis = 1)
    st.subheader('The models translate the data to an embedding space which is a numerical space.')
    st.subheader('The embeddings are fed to sophsticated models such as Matrix Factorisation and Hybrid Models to generate the recommendations.')
    desired_label = st.selectbox('Visualise Embedding Space:', ['Select Space', 'Products', 'Users'])
    space = embed_inner(desired_label,i_e,u_e)
    if desired_label == 'Products' or desired_label == 'Users':
        st.plotly_chart(space)
    st.markdown('_Please wait, it takes time to generate embeddings for the first time before they are cached._')

@st.cache
def embed_inner(desired_label,i_e,u_e):
    if desired_label == 'Select Space':
        print('do nothing')
    elif desired_label == 'Products':
        item_tsne = TSNE(perplexity=30).fit_transform(i_e)
        tsne_df = pd.DataFrame(item_tsne, columns=["tsne_1", "tsne_2"])
        tsne_df["item_id"] = np.arange(item_tsne.shape[0])
        all_ratings = pd.read_csv("all_ratings copy.csv")
        tsne_df = tsne_df.merge(all_ratings.reset_index())

        space = px.scatter(tsne_df, x="tsne_1", y="tsne_2",
                   color="rating",
                   hover_data=["item_id",
                               "products", "popularity"])
        return space
    elif desired_label == 'Users':
        user_tsne = TSNE(perplexity=30).fit_transform(u_e)
        user_tsne_df = pd.DataFrame(user_tsne, columns=["tsne_1", "tsne_2"])
        user_tsne_df["user_id"] = np.arange(user_tsne.shape[0])
        all_ratings = pd.read_csv("all_ratings copy.csv")
        user_tsne_df = user_tsne_df.merge(all_ratings.reset_index())
        space = px.scatter(user_tsne_df, x="tsne_1", y="tsne_2",
                   color="rating",
                   hover_data=["user_id", "products",
                               "item_id", "popularity"])

        return space



def eda():
    data = load_data()
    if st.checkbox('Show Raw Data'):

        st.subheader('Raw data')
        #st.write(data)
        st.dataframe(data[0:1000].style.highlight_max(axis=0))
    #st.subheader('Raw Data')
    #st.write(data)
    pop_prods = pd.read_csv('pop_prods.csv')
    pop_prods = pop_prods[0:35]
    st.subheader('Product popularity at a glance')
    interval = alt.selection_interval()
    st.write(alt.Chart(pop_prods[0:12]).mark_point(opacity=0.8,
    strokeWidth=20).encode(
    x=alt.X('Count', bin = True, axis=alt.Axis(title='Count')),
    color=alt.condition(interval, 'product', alt.value('lightgray')),
    tooltip='product'
    ).interactive().properties(height = 100, width = 600, selection = interval))
    #st.markdown('_Select an area from the chart above_')
    st.subheader('Distribution of ratings')
    #hist_values = np.histogram(data['rating'], bins=6, range=(1,6))[0]
    #st.bar_chart(hist_values)
    #hist = alt.Chart(data['rating']).mark_bar().encode(
    #    alt.X("rating:Q", bin=True),
    #    y='count()',
    #)
    #st.write(hist)

    y = data.groupby(['rating'])['item_id'].count()
    x = [1,2,3,4,5]
    dx = pd.DataFrame({
    'count': y,
    'rating': x,
    })
    st.write(alt.Chart(dx).mark_bar().encode(
    x=alt.X('rating',axis=alt.Axis(title='Rating'), sort=None),
    y='count',
    ).interactive().properties(height= 400, width = 600))

    #hist = alt.Chart(data).mark_bar().encode(
    #    x=alt.X('rating', bin = True, sort = None),
    #    y='count()'
    #).properties(height= 400, width = 600)
    #st.write(hist)
    #st.write(hist)
    #st.histogram(hist)

    #print(data['rating'].describe())
    #plt.hist(data['rating'], bins=5)
    #st.pyplot()
    #st.bar_chart(np.histogram(data['rating']))

    #rating_to_filter = st.slider('rating', 0, 5, 2)
    #filtered_data = data[data['rating'] <= rating_to_filter]
    #st.subheader('Most popular products' % rating_to_filter)
    #st.map(filtered_data)
    st.subheader('Detailed popularity ')
    #x = st.slider(data['rating'])
    desired_label = st.selectbox('Filter to:', ['Users', 'Products'])
    if desired_label == 'Users':
        pop_users = pd.read_csv('pop_users.csv')
        pop_users = pop_users[0:35]
        #st.write(pop_users)
        bars = alt.Chart(pop_users).mark_bar(color='#4db6ac').encode(
            x=alt.X('Count:Q', axis=alt.Axis(title='Total Products Rated')),
            y=alt.Y('userID:O',axis=alt.Axis(title='User ID'),
                    sort=alt.EncodingSortField(
                        field="Users",  # The field to use for the sort
                        order="descending"  # The order to sort in
                        )
                    )
        ).properties(height=800, width = 600)
        st.write(bars)
    else:

        #st.write(pop_users)
        bars = alt.Chart(pop_prods).mark_bar(color='#4db6ac').encode(
            x=alt.X('Count:Q', axis=alt.Axis(title='# Users rated the products high')),
            y=alt.Y('product:O', axis=alt.Axis(title='Products'),
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
    st.markdown('_Models with lower train and validation loss are desirable_')

if __name__ == "__main__":
    main()

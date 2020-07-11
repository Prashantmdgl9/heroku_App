import streamlit as st
import glob
import pickle
from bokeh.plotting import figure, output_file, show, output_notebook,curdoc
from bokeh.models import ColumnDataSource, Plot, LinearAxis, Grid, HoverTool, Legend
from bokeh.models.glyphs import MultiLine
from bokeh.palettes import Category10
from bokeh.models import Range1d


def main():
    files = []
    for file in glob.glob("models/histories/*pkl"):
        files.append(file.rsplit('/')[2])
    page = st.sidebar.selectbox("App Selections", ["Model Results", "EDA"])
    if page == "Model Results":
        st.header("Exploring various models")
        visualize_data(files)
    elif page == "WIP":
        st.header("Work in Progress")

def visualize_data(files):

    models_to_select = []
    for file in files:
        models_to_select.append(file)

    models = st.sidebar.multiselect("Which models do you want to select", models_to_select)
    #y_lim = st.sidebar.selectbox("Choose your y_limit: ", np.arange(0, 10, 0.5))
    y_lim = st.sidebar.slider("Choose your y_limit: ", min_value = 0, max_value = 10, value = 5, step = 1)

    xs = []
    ys = []

    x_p = [0, 5, 10, 15, 20, 25, 30, 35]
    for val in models:
        with open('models/histories/' + val , 'rb') as f:
            load_f = pickle.load(f)
            ys.append(load_f['loss'])
            xs.append(x_p)
            ys.append(load_f['val_loss'])
            xs.append(x_p)

    p = figure(plot_width=750, plot_height=800, title = "Progressive loss for various models")
    p.xaxis.axis_label = "Epochs"
    p.yaxis.axis_label = "Loss"

    label_list = []
    for file in models:
        label_list.append('loss_'+ file)
        label_list.append('val_loss_' + file)

    data = {'xs': xs,
            'ys': ys,
            'labels': label_list,
            'colors': Category10[2* len(files)]}

    source = ColumnDataSource(data)
    p.multi_line(xs='xs', ys='ys', legend = 'labels', color = 'colors', source=source, line_width = 4)
    p.y_range = Range1d(0, y_lim)
    #st.bokeh_chart(p)
    st.write(p)


if __name__ == "__main__":
    main()

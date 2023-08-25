from typing import Container
from xml.sax.handler import feature_external_ges
import streamlit as st
import pandas as pd
import plotly.express as px
from herows import *
import re
from collections import defaultdict
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
import numpy as np
from PIL import Image



# Initialize Streamlit app
run_herows()

st.set_page_config(page_title="HEAowS", 
                   page_icon=":bar_chart:",
                   layout="wide")

with open("style.css") as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Load your DataFrame (cached using st.cache)
@st.cache_data
def load_data_HEA_exp():
    return pd.read_excel(io="data_experimentale_herows.xlsx", engine='openpyxl', sheet_name='Sheet1')

data_exp = load_data_HEA_exp()


@st.cache_data
def load_data_HEA_calculated():
    return pd.read_excel(io="new herows data.xlsx", engine='openpyxl', sheet_name='Sheet1')

df = load_data_HEA_calculated()

@st.cache_data
def load_data_Decile():
    return pd.read_excel(io="Decile_data.xlsx", engine='openpyxl', sheet_name='Sheet1')

Decile = load_data_Decile()
sorted_decile = Decile.sort_values(by="HERowS_Score")


@st.cache_data
def load_data_Decile_exp():
    return pd.read_excel(io="Decile_data_experimentale.xlsx", engine='openpyxl', sheet_name='Sheet1')

Decile_exp = load_data_Decile_exp()



# X and Y selection using containers and caching
container1 = st.container()


col11,col12,col13,col14,col15  = container1.columns([0.3,0.15,0.15,0.15,0.15])
#col121 , col122 = col12.columns(2)
col11.markdown('### Calculate the properties 👇')


alloy_input = col11.text_input('Please enter the alloy as the following form : `An1Bn2Cn3`'  , 'Au1AlOs')


v_fco2_avg = fCO2(alloy_input)
v_fe_avg = fE(alloy_input)
v_hhi = hhi(alloy_input)
v_esg = esg(alloy_input)
v_sr = Supply_risk(alloy_input)
v_p_avg = p_avg(alloy_input,100)
v_r_avg = r_avg(alloy_input,100)
v_c_avg = c_avg(alloy_input)
v_p_max = p_max(alloy_input)
v_r_max = r_max(alloy_input)
v_c_max = c_max(alloy_input)

decile_co2 = assign_decile(v_fco2_avg,"FCO2",df)
decile_fe = assign_decile(v_fe_avg,"FCO2",df)
decile_hhi = assign_decile(v_hhi,"HHI",df)
decile_esg = assign_decile(v_esg,"ESG",df)
decile_sr = assign_decile(v_sr,"SR",df)
decile_p_avg = assign_decile(v_p_avg,"P_avg",df)
decile_r_avg = assign_decile(v_r_avg,"R_avg",df)
decile_c_avg = assign_decile(v_c_avg,"C_avg",df)
decile_p_max = assign_decile(v_p_max,"P_max",df)
decile_r_max = assign_decile(v_r_max,"R_max",df)
decile_c_max = assign_decile(v_c_max,"C_max",df)

herows_score = decile_co2 + decile_sr +decile_p_avg + decile_r_avg + decile_c_avg
decile_herows_score = assign_decile(herows_score,"HERowS_Score",Decile)
@st.cache_resource
def h():
    v_fco2_avg = fCO2(alloy_input)
    v_fe_avg = fE(alloy_input)
    v_hhi = hhi(alloy_input)
    v_esg = esg(alloy_input)
    v_sr = Supply_risk(alloy_input)
    v_p_avg = p_avg(alloy_input,100)
    v_r_avg = r_avg(alloy_input,100)
    v_c_avg = c_avg(alloy_input)
    v_p_max = p_max(alloy_input)
    v_r_max = r_max(alloy_input)
    v_c_max = c_max(alloy_input)

    decile_co2 = assign_decile(v_fco2_avg,"FCO2",df)
    decile_fe = assign_decile(v_fe_avg,"FCO2",df)
    decile_hhi = assign_decile(v_hhi,"HHI",df)
    decile_esg = assign_decile(v_esg,"ESG",df)
    decile_sr = assign_decile(v_sr,"SR",df)
    decile_p_avg = assign_decile(v_p_avg,"P_avg",df)
    decile_r_avg = assign_decile(v_r_avg,"R_avg",df)
    decile_c_avg = assign_decile(v_c_avg,"C_avg",df)
    decile_p_max = assign_decile(v_p_max,"P_max",df)
    decile_r_max = assign_decile(v_r_max,"R_max",df)
    decile_c_max = assign_decile(v_c_max,"C_max",df)

    herows_score = decile_co2 + decile_sr +decile_p_avg + decile_r_avg + decile_c_avg
    decile_herows_score = assign_decile(herows_score,"HERowS_Score",Decile)
  

columns = col11.columns([0.375,0.25,0.375])
button_pressed = columns[1].button('Calculate')

if button_pressed:
    h()



st.write(
    """
    <style>
    [data-testid="stMetricDelta"] svg {
        display: none;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


st.write(
    """
    <style>
    [data-testid="metric-container"]  {
        background-color: #FFFFFF;
        border: 1px solid #CCCCCC;
        padding: 5% 5% 5% 10%;
        border-radius: 5px;
        
        border-left: 0.5rem solid #9AD8E1 !important;
        box-shadow: 0 0.15rem 1.75rem 0 rgba(58, 59, 69, 0.15) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.write(
    """
    <style>
    [data-testid="stMetricValue"]  {
        font-size: 30px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.write(
    """
    <style>

    </style>
    """,
    unsafe_allow_html=True,
)

st.write(
    """
    <style>
    .row-widget.stButton{
        justify-content: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)




col12.metric("Carbon Footprint (g CO2/mol)", round(v_fco2_avg,2), bijection_decile_value(decile_co2))
col12.metric("Energy Footprint (MJ/mol)", round(v_fe_avg,2), bijection_decile_value(decile_fe))
col12.metric("Herfindahl-Hirschman Index", round(v_hhi,2), bijection_decile_value(decile_hhi))

col13.metric("ESG risk", round(v_esg,2), bijection_decile_value(decile_esg))
col13.metric("Supply risk", round(v_sr,2), bijection_decile_value(decile_sr))
col13.metric("Average Companionality",round(v_c_avg,2), bijection_decile_value(decile_c_avg))

col14.metric("Maximal Companionality", round(v_c_max,2), bijection_decile_value(decile_c_avg))
col14.metric("Average Production (t/year)", round(v_p_avg,2), bijection_decile_value(decile_p_avg))
col14.metric("Maximal Production (t/year)", round(v_p_max,2), bijection_decile_value(decile_p_max))

col15.metric("Average Reserves (tons)", round(v_r_avg,2), bijection_decile_value(decile_r_avg))
col15.metric("Maximal Reserves (tons)", round(v_r_max,2), bijection_decile_value(decile_r_avg))
col15.metric("HERowS Score", round(herows_score,2),bijection_decile_value(decile_herows_score))





# col121.write(f"fco2_avg: {round(v_fco2_avg,3)} ({decile_co2})")
# col121.write(f"fe_avg: {round(v_fe_avg,3)} ({decile_fe})")
# col121.write(f"hhi: {round(v_hhi,3)} ({decile_hhi})")
# col121.write(f"esg: {round(v_esg,3)} ({decile_esg})")
# col121.write(f"sr: {round(v_sr,3)} ({decile_sr})")
# col121.write(f"p_avg: {round(v_p_avg,3)} ({decile_p_avg})")

# col122.write(f"r_avg: {round(v_r_avg,3)} ({decile_r_avg})")
# col122.write(f"c_avg: {round(v_c_avg,3)} ({decile_c_avg})")
# col122.write(f"p_max: {round(v_p_max,3)} ({decile_p_max})")
# col122.write(f"r_max: {round(v_r_max,3)} ({decile_r_avg})")
# col122.write(f"c_max: {round(v_c_max,3)} ({decile_c_avg})")
# col122.write(f"herows_score: {round(herows_score,3)}")

container_img = col11.container()

image = Image.open('img.jpg')

container_img.image(image, caption='Only colored elements are accepted')

st.markdown('----')

column_names = ['index', 'HHI', 'ESG', 'SR', 'C_avg', 'P_avg','R_avg', 'C_max','P_max', 'R_max', 'FE', 'FCO2']
pt_data = [[alloy_input,v_hhi,v_esg,v_sr,v_c_avg,v_p_avg,v_r_avg,v_c_max,v_p_max,v_r_max,v_fe_avg,v_fco2_avg]]
df_pt = pd.DataFrame(pt_data, columns=column_names)


#################-----> CONTAINER 2 <-----#################


container2 = st.container()
col21,col22 = container2.columns([0.7, 0.3] , gap = "large")


col211,col212 = col21.columns(2)

varia = {"Carbon Footprint (g CO2/mol)" : "FCO2",  
            "Energy Footprint (MJ/mol)" : "FE", 
            "Herfindahl-Hirschman Index" : "HHI",   
            "ESG risk" : "ESG", 
            "Supply risk" : "SR", 
            "Average Companionality" : "C_avg",
            "Maximal Companionality" : "C_max", 
            "Average Production (t/year)" : "P_avg" ,
            "Maximal Production (t/year)" : "P_max",
            "Average Reserves (tons)" : "R_avg",
            "Maximal Reserves (tons)" : "R_max"
            }


X_scatter_plot = col211.selectbox(
    "Select the X axis",
    options=list(varia.keys()),
    key="selectbox_X"  # Key helps Streamlit identify the widget
)

Y_scatter_plot = col212.selectbox(
    "Select the Y axis",
    options=list(varia.keys()),
    key="selectbox_Y"  # Key helps Streamlit identify the widget
)




def create_scatter_plot(X_scatter_plot, Y_scatter_plot):
        scatter_plot = px.scatter(df, x=varia[X_scatter_plot], y=varia[Y_scatter_plot], hover_name='index')
        scatter_plot.update_layout(title=f"<b>Scatter Plot: {X_scatter_plot} vs {Y_scatter_plot}</b>",
                                xaxis_title=X_scatter_plot,
                                yaxis_title=Y_scatter_plot,
                                
                                title_x=0.2,
                                width=800,
                                height=600)
        scatter_plot.update_traces(marker=dict(
            color='#0593A2'))
        return scatter_plot


@st.cache_resource   
def create_scatter_plot_exp(X_scatter_plot, Y_scatter_plot):

    
    scatter_plot_2 = px.scatter(data_exp, x=varia[X_scatter_plot], y=varia[Y_scatter_plot], hover_name='index')
    scatter_plot_2.update_traces(marker=dict(
        color='#0513A2'))
    scatter_plot_2_trace = scatter_plot_2.data[0]

    return scatter_plot_2_trace

@st.cache_resource  
def create_scatter_plot_pt(X_scatter_plot, Y_scatter_plot,sp):

    
    scatter_plot_2 = px.scatter(df_pt, x=varia[X_scatter_plot], y=varia[Y_scatter_plot], hover_name='index')
    scatter_plot_2.update_traces(marker=dict(
        color='#FF0000'))
    X= float(df_pt[varia[X_scatter_plot]])
    Y= float(df_pt[varia[Y_scatter_plot]])
    sp.add_shape(type='line', x0=X, x1=X, y0=0, y1=Y, line=dict(color='red', dash='dash'))
    sp.add_shape(type='line', x0=0, x1=X, y0=Y, y1=Y, line=dict(color='red', dash='dash'))
    scatter_plot_2_trace = scatter_plot_2.data[0]

    return scatter_plot_2_trace






container21 = col21.container()
container22 = col21.container()
col_cont21_1,col_cont21_2,col_cont21_3,col_cont21_4 = container22.columns(4)

log_x_agree = col_cont21_3.checkbox('Logarithmize x-axis')
log_y_agree = col_cont21_4.checkbox('Logarithmize y-axis')


scatter_agree_exp = col_cont21_1.checkbox('Show experimental data')
scatter_agree_point = col_cont21_2.checkbox('Show calculated point')

scatter_plot = create_scatter_plot(X_scatter_plot, Y_scatter_plot)

if log_x_agree :
    scatter_plot.update_xaxes(type='log')

if log_y_agree :
    scatter_plot.update_yaxes(type='log')


if scatter_agree_exp == True :
    scatter_plot.add_trace(create_scatter_plot_exp(X_scatter_plot, Y_scatter_plot))



if scatter_agree_point == True :
    scatter_plot.add_trace(create_scatter_plot_pt(X_scatter_plot, Y_scatter_plot,scatter_plot))


container21.plotly_chart(scatter_plot,use_container_width=True)



l_pour = {"<10%" : 1, 
    str('10% - 20%')  :2,
    str('20% - 30%') : 3,
    str('30% - 40%') :4,
    str('40% - 50%'): 5,
    str('50% - 60%') : 6,
    str('60% - 70%') : 7,
    str('70% - 80%') : 8,
    str('80% - 90%') : 9,
    "<90%" : 10}

decile_bij = {"Average Companionality" : "C_avg_Decile" ,
              "Carbon Footprint" : "FCO2_avg_Decile",    
              "Supply risk" : "SR_Decile",    
              "Average Production" : "P_avg_Decile",
              "Average Reserves" : "R_avg_Decile"}



col221 , col222 = col22.columns(2)

var_circular_barplot_var_decile = col221.selectbox(
    "Select a propertie",
    options=list(decile_bij.keys()),
    key="selectbox_var_circular_barplot_var_decile"  # Key helps Streamlit identify the widget
)

decile_circular_barplot_var_decile = col222.selectbox(
    "Select a range of %",
    options= list(l_pour.keys()),
    key="selectbox_decile_circular_barplot_var_decile"  # Key helps Streamlit identify the widget
)

@st.cache_resource
def circular_barplot_var_decile(var, decile):
    element_counts_regular = {}
    a = []
    b = []
    element_counts = {}

    selected_data = Decile[["index", var]][Decile[var] == decile]

    element_counts = defaultdict(int)

    for index_value in selected_data["index"]:
        elements = index_value.split("-")
        for element in elements:
            element_counts[element] += 1
    element_counts_regular = dict(element_counts)

    a = list(element_counts_regular.keys())
    b = list(element_counts_regular.values())
    fig = go.Figure()
    fig.add_trace(go.Barpolar(r=b, theta=a, text=a, hoverinfo='text+r'))
    fig.update_layout(
        polar=dict(radialaxis=dict(showticklabels=False, ticks=''), angularaxis=dict(direction="clockwise"))
    )
    
    return fig  # Return the figure object

cirular_bar1 = circular_barplot_var_decile(decile_bij[var_circular_barplot_var_decile], l_pour[decile_circular_barplot_var_decile])
col22.plotly_chart(cirular_bar1 ,use_container_width=True)
col22.markdown("######    In this plot, we aim to visualize the count of primary elements within the chosen percentage range for the selected propertie.")


st.markdown('----')



#################-----> CONTAINER 3 <-----#################


container3 = st.container()
col31,col32 = container3.columns([0.3, 0.7], gap = "large" )


col311,col312 = col31.columns(2 , gap = "large")

best_worst = col311.selectbox(
    "Select best or worst",
    options=["best",'worst'],
    key="selectbox_best_worst"  # Key helps Streamlit identify the widget
)

n_best_worst = col312.number_input("Pick the number of alloys",1,30000)



@st.cache_resource
def circular_barplot_best_600(etat , n):
    element_counts_regular = {}
    a = []
    b = []
    element_counts = {}
    sorted_decile = Decile.sort_values(by="HERowS_Score")
    if etat == 'best' : 
        selected_rows_10 = sorted_decile.head(n)
    else :
        selected_rows_10 = sorted_decile.tail(n)
    
    element_counts = defaultdict(int)
    for index_value in selected_rows_10["index"]:
        elements = index_value.split("-")  # Diviser la valeur en éléments chimiques
        for element in elements:
            element_counts[element] += 1  # Incrémenter le compteur d'occurrences

    # Convertir le dictionnaire en un dictionnaire régulier (si nécessaire)
    element_counts_regular = dict(element_counts)
    
    a = list(element_counts_regular.keys())
    b = list(element_counts_regular.values())
    fig = go.Figure()
    fig.add_trace(go.Barpolar(r=b,theta=a,text=a,hoverinfo='text+r',))
    fig.update_layout(
    title="Circular Barplot for the best/wordt n alloys",
    title_x = 0.1,
    polar=dict(radialaxis=dict(showticklabels=False, ticks=''),angularaxis=dict(direction="clockwise"),))
    return fig

cirular_bar2 = circular_barplot_best_600(best_worst , n_best_worst)
col31.plotly_chart(cirular_bar2 ,use_container_width=True)

container34 = col32.container()
container33 = col32.container()
c1,c3,c4,c5 = container33.columns((1.5,1,0.9,0.9))

agree_dens = c3.checkbox('Show experimental data' , key ='ddfdf')

@st.cache_resource
def plot_density():
    hist_data = [list(Decile["HERowS_Score"])]
    group_labels = ['Synthetic alloys']
    color = ["#103778"]
    fig = ff.create_distplot(hist_data, group_labels, colors=color, show_rug=False)
    fig.update_layout(
        title="Density Plot of HERowS Scores",
        title_x = 0.3,
        xaxis_title="HERowS Score",
        yaxis_title="Density",
    )

    return fig

@st.cache_resource
def plot_density_exp():
    hist_data = [list(Decile["HERowS_Score"]),list(Decile_exp["HERowS_Score"])]
    group_labels = ['Synthetic alloys','Experimental alloys']
    color = ["#103778",'#FF0000']
    fig = ff.create_distplot(hist_data, group_labels, colors=color, show_rug=False)
    fig.update_layout(
        title="Density Plot of HERowS Scores",
        title_x = 0.3,
        xaxis_title="HERowS Score",
        yaxis_title="Density",
    )

    return fig

density_plot = plot_density()
if agree_dens :
    density_plot = plot_density_exp()
else :
    density_plot = plot_density()

container34.plotly_chart(density_plot ,use_container_width=True  ,use_container_height=True)


st.markdown('----')


#######

########## CNTAINER 4 3#######

container4 = st.container()
col41,col42 = container4.columns([0.7, 0.3], gap = "large" )


container43 = col41.container()
container44 = col41.container()


@st.cache_resource
def cumulative_graph_exp(herows_score,etat):
    density = np.histogram(Decile["HERowS_Score"], bins=30, density=True)
    cumulative_density = np.cumsum(density[0]) * np.diff(density[1])[0]

    density_exp = np.histogram(Decile_exp["HERowS_Score"], bins=30, density=True)
    cumulative_density_exp = np.cumsum(density_exp[0]) * np.diff(density_exp[1])[0]

    # Create cumulative KDE plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=density[1][1:], y=cumulative_density, mode='lines' , name = "Synthetic data") )
    fig.add_trace(go.Scatter(x=density_exp[1][1:], y=cumulative_density_exp, mode='lines',line=dict(color='green') , name='Experimental alloys'))

    # Update layout
    fig.update_layout(
        title="Cumulative KDE of HERowS Score",
        title_x = 0.4,
        xaxis_title="HERowS_Score",
        yaxis_title="Cumulative Density",
    )
    if etat == True :

        fig.add_shape(
            type="line",
            x0=herows_score,
            x1=herows_score,
            y0=0,
            y1=1,
            line=dict(color="red", width=2),
            )
    return fig

@st.cache_resource
def cumulative_graph(herows_score,etat):
    density = np.histogram(Decile["HERowS_Score"], bins=30, density=True)
    cumulative_density = np.cumsum(density[0]) * np.diff(density[1])[0]

    # Create cumulative KDE plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=density[1][1:], y=cumulative_density, mode='lines'))
    # Update layout
    fig.update_layout(
        title="Cumulative KDE of HERowS Score",
        title_x = 0.4,
        xaxis_title="HERowS_Score",
        yaxis_title="Cumulative Density",
    )
    if etat == True :

        fig.add_shape(
            type="line",
            x0=herows_score,
            x1=herows_score,
            y0=0,
            y1=1,
            line=dict(color="red", width=2),
            )

    return fig


d1,d2,d3,d4 = container44.columns((1,1,1,1))

agree_cum = d2.checkbox('Show experimental alloys' , key = "d2")
agree_cum_pt = d3.checkbox('Show calculated point', key = "d3")

if agree_cum :
    cum_plot = cumulative_graph_exp(herows_score,agree_cum_pt)
else :
    cum_plot = cumulative_graph(herows_score,agree_cum_pt)
container43.plotly_chart(cum_plot ,use_container_width=True  ,use_container_height=True)


@st.cache_resource
def circular_barplot_all():
    element_counts_regular = {}
    a = []
    b = []
    element_counts = {}
    element_counts = defaultdict(int)
    for index_value in Decile["index"]:
        elements = index_value.split("-")  # Diviser la valeur en éléments chimiques
        for element in elements:
            element_counts[element] += 1  # Incrémenter le compteur d'occurrences

    # Convertir le dictionnaire en un dictionnaire régulier (si nécessaire)
    element_counts_regular = dict(element_counts)
    
    a = list(element_counts_regular.keys())
    b = list(element_counts_regular.values())
    fig = go.Figure()
    fig.add_trace(go.Barpolar(r=b,theta=a,text=a,hoverinfo='text+r',))
    fig.update_layout(
    title="Circular Barplot for all the 30000 alloys",
    title_x = 0.1,
    polar=dict(radialaxis=dict(showticklabels=False, ticks=''),angularaxis=dict(direction="clockwise"),))
    return fig


all_plot_circular_bar = circular_barplot_all()
col42.plotly_chart(all_plot_circular_bar ,use_container_width=True  ,use_container_height=True)


import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
# 表示  express
import plotly


# filename='D:\\py\\data\\1962_2006_walmart_store_openings.csv'
# filename='1962_2006_walmart_store_openings.csv'
filename=r"G:\file\学校\可视化\大作业\COVID-19\COVID-19-Data-master\US\County_level_summary\US_County_summary_covid19_confirmed.csv"

walmart_loc_df = pd.read_csv(filename)
walmart_loc_df.head()
#Plot the scatter plot


fig = go.Figure(data=go.Scattergeo(
        lon = walmart_loc_df['LON'], # column containing longitude information of the locations to plot
        lat = walmart_loc_df['LAT'], # column containing latitude information of the locations to plot
        text = walmart_loc_df['STREETADDR'], # column containing value to be displayed on hovering over the map
        mode = 'markers' # a marker for each location
        ))

fig.update_layout(
        title = 'Walmart stores across the USA',
        geo_scope='usa',
    )

fig.show()

plotly.offline.plot(fig, filename='Walmart stores across the USA_Scatter Chart.html')



#Let’s first compute the number of Walmart stores per state. 
walmart_stores_by_state = walmart_loc_df.groupby('STRSTATE').count()['storenum'].reset_index().rename(columns={'storenum':'NUM_STORES'})
walmart_stores_by_state.head()

#For generating the bubble plots, we will use the plotly express module and the scatter_geo function. Notice how the locations parameter is set to the name of column which contains state codes, and the size parameter is set to the feature NUM_STORES.
#为了生成气泡图，我们将使用plotly express模块和散射函数。
# 请注意，locations参数如何设置为包含状态代码的列的名称，size参数如何设置为feature NUM_STORES。
fig = px.scatter_geo(walmart_stores_by_state, 
                    locations="STRSTATE", # name of column which contains state codes
                    #包含州代码的列的名称
                    size="NUM_STORES", # name of column which contains aggregate value to visualize
                    locationmode = 'USA-states',
                    hover_name="STRSTATE",
                    size_max=45)
                    
fig.update_layout(
    # add a title text for the plot
    title_text = 'Walmart stores across states in the US-Bubble Chart',
    # limit plot scope to USA
    geo_scope='usa'
)

fig.show()
plotly.offline.plot(fig, filename='Walmart stores across states in the US-Bubble Chart.html')

# 不能运行
# raise KeyError(key) from err
# KeyError: 'LON'

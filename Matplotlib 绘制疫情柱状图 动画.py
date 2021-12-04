
import matplotlib.pyplot as plt
import pandas as pd
# matplolib.animation.FuncAnimation
import  matplotlib.animation as ani

df = pd.read_csv('a.csv', delimiter=',', header='infer')
df_interest = df.loc[df['Country/Region'].isin(['United Kingdom', 'US', 'Italy', 'Germany'])& df['Province/State'].isna()]
df_interest.rename(index=lambda x: df_interest.at[x, 'Country/Region'], inplace=True)
df1 = df_interest.transpose()
df1 = df1.drop(['Province/State', 'Country/Region', 'Lat', 'Long'])
df1 = df1.loc[(df1 != 0).any(1)]
df1.index = pd.to_datetime(df1.index)


fig = plt.figure(figsize=(9,16))


def buildbarh(i=int):
    iv = min(i, len(df1.index)-1)
    objects = df1.max().index
    y_pos = np.arange(len(objects))
    performance = df1.iloc[[iv]].values.tolist()[0]

    plt.barh(y_pos, performance, align='center', color=['red', 'green', 'blue', 'orange'])
    plt.subplots_adjust(left=0.2)
    plt.yticks(y_pos, objects)
    plt.xlabel('Deaths')
    plt.ylabel('Countries')

# getmepie func
animator = ani.FuncAnimation(fig, getmepie, interval = 200)
plt.show()

def buildbar(i=int):
    iv = min(i, len(df1.index)-1)
    objects = df1.max().index
    y_pos = np.arange(len(objects))
    performance = df1.iloc[[iv]].values.tolist()[0]

    plt.bar(y_pos, performance, align='center', color=['red', 'green', 'blue', 'orange'])
    plt.subplots_adjust(left=0.2)
    plt.xticks(y_pos, objects)
    plt.ylabel('Deaths')
    plt.xlabel('Countries')
    plt.title('Deaths per Country \n' + str(df1.index[iv].strftime('%y-%m-%d')))
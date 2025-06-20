import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

#Check
# 1
df = pd.read_csv('medical_examination.csv')

# 2
bmi = df['weight'] / ((df['height'] / 100) ** 2)
df['overweight'] = (bmi > 25).astype(int)

# 3
df['cholesterol'] = (df['cholesterol'] > 1).astype(int)
df['gluc'] = (df['gluc'] > 1).astype(int)

# 4
def draw_cat_plot():
    # 5
    df_cat = df.copy()
    df_cat = pd.melt(
        df_cat,
        id_vars=['cardio'],  # manter esta coluna fixa
        value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight']
    )

    # 6
    df_cat = df_cat.groupby(['cardio', 'variable', 'value']).size().reset_index(name='total')
    

    # 7

    plot = sns.catplot(
        x='variable',
        y='total',
        hue='value',
        col='cardio',
        kind='bar',
        data=df_cat
    )


    # 8
    fig = plot.fig


    # 9
    fig.savefig('catplot.png')
    return fig


# 10
def draw_heat_map():
    # 11
    df_heat = df.copy()

    #Cleaning 1
    df_heat = df_heat[(df_heat['ap_lo'] <= df_heat['ap_hi']) &
    (df_heat['height'] >= df_heat['height'].quantile(0.025)) &
    (df_heat['height'] <= df_heat['height'].quantile(0.975)) &
    (df_heat['weight'] >= df_heat['weight'].quantile(0.025)) &
    (df_heat['weight'] <= df_heat['weight'].quantile(0.975))]

    # 12
    corr = df_heat.corr().round(1)

    # 13
    mask = np.triu(np.ones_like(corr, dtype=bool))



    # 14
    fig, ax = plt.subplots(figsize=(12, 10))

    # 15
    sns.heatmap(
    corr,
    mask=mask,             # Aplica a máscara à parte superior
    annot=True,            # Mostra os valores dentro das células
    fmt=".1f",             # Mostra os valores com uma casa decimal
    center=0,              # Centro da escala de cores
    square=True,           # Mantém as células quadradas
    linewidths=0.5,        # Linhas de separação entre as células
    cbar_kws={"shrink": 0.5}  # Ajusta o tamanho da barra lateral
    )


    # 16
    fig.savefig('heatmap.png')
    return fig

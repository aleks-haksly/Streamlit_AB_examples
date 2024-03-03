import pandas as pd
import plotly.graph_objects as go
import numpy as np
import os
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
import scipy
import pingouin as pg
import statsmodels.formula.api as smf
from statsmodels.stats.api import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import warnings

warnings.filterwarnings('ignore')
st.set_page_config(page_title='Множественные сравнения')

st.header('Множественные сравнения на синтетических данных')
"""
Посмотрим имеющийся у нас DataFrame, содержащий информацию о количестве лайков с разбивкой по цвету кнопок.

"""

data_file_path = os.path.join(os.path.dirname(__file__), "5 post_likes.csv")
df = pd.read_csv(data_file_path)

st.dataframe(df, use_container_width=True, height=210)
"""
Построим Boxplot, чтобы посмотреть на распределение лайков в зависисмости от цвета кнопок.

"""
fig = px.box(df, color="button", y="likes", color_discrete_sequence=["red", "green", "blue"])
fig.update_layout(
    autosize=True,
    margin=dict(
        l=0,
        r=0,
        b=0,
        t=0,
        pad=0
    ))
st.plotly_chart(fig, use_container_width=True)
st.header('ANOVA', anchor='anva')
st.write(
    'Выполним однофакторный дисперсионный анализ (ANOVA) чтобы проверить, являются ли наблюдаемые различия между тремя группами статистически значимыми.')

st.write("""
#### С помощью модуля *scipy* ####
```python
scipy.stats.f_oneway(*list(zip(*df.groupby('button')['likes']))[1])""")
statistic, pvalue = scipy.stats.f_oneway(*list(zip(*df.groupby('button')['likes']))[1])
col1, col2 = st.columns(2)
col1.metric('statistic', round(statistic, 3))
col2.metric('p-value', "{:e}".format(pvalue))
st.write("""
#### С помощью модуля *statsmodels* ####
```python
model = smf.ols("likes ~ C(button)", data=df).fit()
anova_lm(model)
""")
model = smf.ols(formula="likes ~ C(button)", data=df).fit()
st.table(anova_lm(model))
st.write("""
#### С помощью модуля *pingouin* ####
```python
pg.anova(data=df, dv="likes", between="button")
""")
st.table(pg.anova(data=df, dv="likes", between="button"))
st.header("Критерий Краскела — Уоллиса (непараметрический аналог однофакторного ANOVA)")
"""
```python
pg.kruskal(data=df, between='button', dv='likes')
"""
st.write(pg.kruskal(data=df, between='button', dv='likes'))

st.header('Доверительные интервалы')
st.write('Построим 95% доверительные интервалы для средних в каждой группе')

fig = go.Figure(data=go.Scatter(
    x=['red', 'green', 'blue'],
    y=list(map(np.mean, list(zip(*df.groupby('button')['likes']))[1]))[::-1],
    error_y=dict(
        type='data',  # value of error bar given in data coordinates
        array=[_ / 10 * 1.96 for _ in map(np.std, list(zip(*df.groupby('button')['likes']))[1])][::-1],
        visible=True)
))
fig.update_layout(
    autosize=True,
    margin=dict(
        l=0,
        r=0,
        b=0,
        t=0,
        pad=0
    ))
fig.update_xaxes(type='category')
st.plotly_chart(fig, use_container_width=True)

st.header('Проверим нормальность распределения данных')
st.subheader("Проверка на нормальность с помощью теста Шапиро-Уилка")
"""
Этот тест лучше не применять на больших выборок во избежание ошибки 2 рода
```python
for _ in list(zip(*df.groupby('button')['likes']))[1]:
    print(scipy.stats.shapiro(_))
"""
for _ in list(zip(*df.groupby('button')['likes']))[1]:
    st.text(scipy.stats.shapiro(_), )
st.subheader("Проверка на нормальность с помощью библтотеки *scipy* (D’Agostino - Pearson)")
"""
Подходит для больших выборок
```python
for _ in list(zip(*df.groupby('button')['likes']))[1]:
    print(scipy.stats.normaltest(_))
"""
for _ in list(zip(*df.groupby('button')['likes']))[1]:
    st.text(scipy.stats.normaltest(_))

st.subheader("Проверка на нормальность с помощью библиотеки *pinguin*")
"""```python
for _ in list(zip(*df.groupby('button')['likes']))[1]:
    print(pg.normality(_, method='**method**'))
"""
option = st.selectbox(
    "Выберете метод проверки нормальности",
    ('shapiro', 'normaltest', 'jarque_bera'),
    index=0,
    placeholder="Выбран метод...",
)
test = []
for _ in list(zip(*df.groupby('button')['likes']))[1]:
    test.append(pg.normality(_, method=option)[['pval']].T)
result = pd.concat(test, axis=1)
result.columns = ['red', 'green', 'blue']
st.table(result)

st.header("Построение QQ-plot")
labels = ["red buttons", "green buttons", "blue buttons"]
option = st.radio(
    "Выберите группу для построения графика",
    labels,
    index=1)
fig, ax = plt.subplots()
scipy.stats.probplot(list(df.groupby('button')['likes'])[labels.index(option)][1], plot=ax)
st.pyplot(fig, use_container_width=True, clear_figure=True)

st.header('Тестируем на гомогенность дисперсии')
"""
```python
scipy.stats.levene(*list(zip(*df.groupby('button')['likes']))[1])

"""
st.text(scipy.stats.levene(*list(zip(*df.groupby('button')['likes']))[1]))

"""

```python
pg.homoscedasticity(df, dv='likes', group='button')
"""
st.table(pg.homoscedasticity(df, dv='likes', group='button'), )
"""
*Дисперсия в разных груаппах не гомогенна, поэтому применим:*
ANOVA Вэлча 
```python
pg.welch_anova(df, dv='likes', between='button')
"""
st.table(pg.welch_anova(df, dv='likes', between='button'))
"""
или 
```python
scipy.stats.ttest_ind(equal_var=False)```
"""
"""
*Если ANOVA показала нам, что разница между группами существует, то переходим к исследованию того, между какими в точности группами есть разница*
***
"""
st.header("Попарные сравнения")
st.markdown(
    """
1. Попарные сравнения c t-test
* Без поправки (не надо так) 
pg.pairwise_tests(data=, dv=' ', between=' ') p-unc
* С поправкой Бонферонни: pg.pairwise_tests(data=, dv=' ', between=' ', padjust='bonf') p-unc --> p-corr (самая консервативная)
* С поправкой Холма:  pg.pairwise_tests(data=, dv=' ', between=' ', padjust='holm') p-unc --> p-corr (менее консервативная)
* С поправкой Бенжамини-Хохберга:  padjust='fdr_bh' - наименне консервативная
* И т.д.
2. Для одинаковых диперсий в группах - попарные сравнения Тьюки
* pg.pairwise_tukey(data=, dv=' ', between=' ')
4. Для различающихся дисперсий в группах Геймса-Хоувелла
* pg.pairwise_gameshowell(data=, dv=' ', between=' ')
5. Реализация с statsmodels:
* pairwise_tukeyhsd(endog=df.likes, groups=df.button).summary()
* MultiComparison(data=df.likes, groups=df.button).tukeyhsd().summary()
***
""")
"""```python
pg.pairwise_ttests(df, dv='likes', between='button')"""
st.table(pg.pairwise_ttests(df, dv='likes', between='button'))

"""
```python
pg.pairwise_ttests(df, dv='likes', between='button', padjust='bonf')
"""
st.table(pg.pairwise_ttests(df, dv='likes', between='button', padjust='bonf'))

"""
```python
pg.pairwise_tests(df, dv='likes', between='button', padjust='fdr_bh')
"""
st.table(pg.pairwise_tests(df, dv='likes', between='button', padjust='fdr_bh'))

"""
```python
pg.pairwise_tukey(df, dv='likes', between='button')
"""
st.table(pg.pairwise_tukey(df, dv='likes', between='button'))

"""```python
pg.pairwise_tukey(df, dv='likes', between='button')
"""
st.table(pg.pairwise_tukey(df, dv='likes', between='button'))

"""
```python
pg.pairwise_gameshowell(df, dv='likes', between='button')
"""
st.table(pg.pairwise_gameshowell(df, dv='likes', between='button'))

"""
```python
pairwise_tukeyhsd(endog=df.likes, groups=df.button).summary()
"""
st.table(pairwise_tukeyhsd(endog=df.likes, groups=df.button).summary())

st.header('2 way ANOVA')
data_file_path = os.path.join(os.path.dirname(__file__), "5 ads_clicks.csv")
df = pd.read_csv(data_file_path)
st.dataframe(df, use_container_width=True, height=210)
st.subheader('Ads types by ages vs clicks')
fig = px.box(df, x="ads", y="clicks", color="age_group")
fig.update_layout(
    autosize=True,
    margin=dict(
        l=0,
        r=0,
        b=0,
        t=0,
        pad=0
    ))
st.plotly_chart(fig, use_container_width=True)

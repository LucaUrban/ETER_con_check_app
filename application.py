import streamlit as st
import time
import base64
import os
from urllib.parse import quote as urlquote
from urllib.request import urlopen
import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype
import json
import random
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from plotly.colors import n_colors
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.stattools import medcouple
import math
import scipy.stats as stats
import pymannkendall as mk
import csv

st.title("Visual Information Quality Environment")
st.write("In this part you can upload your csv file either dropping your file or browsing it. Then the application will start showing all of the charts for the Dataset. " +
         "To change the file to be analyzed you have to refresh the page.")
uploaded_file = st.file_uploader("Choose a file")
demo_data_radio = st.radio("What is the dataset you want to import:", ('Demo datset', 'ETER Dataset', 'Another dataset'))

if demo_data_radio == 'Demo datset' or uploaded_file is not None:
    if uploaded_file is not None:
        if demo_data_radio == 'ETER Dataset':
            table = pd.read_csv(uploaded_file, delimiter = ';', decimal = ',')
            
            for col in table.columns:
                if table[col].dtypes == 'O' and not (col.startswith('Flag') or col.startswith('Notes')):
                    table[col] = table[col].apply(lambda x: x.replace(',', '.') if not pd.isna(x) else x)
            
            for token in ['a', 'c', 'm',  'nc','s', 'x', 'xc', 'xr']:
                table.replace({token: np.nan}, inplace = True)
            
            for col in table.columns:
                table[col] = pd.to_numeric(table[col], errors = 'ignore')
        else:
            table = pd.read_csv(uploaded_file)
    else:
        table = pd.read_csv('https://raw.githubusercontent.com/LucaUrban/prova_streamlit/main/eter_ratio_fin_wf.csv')

    # importing all other necessary files
    with urlopen('https://raw.githubusercontent.com/leakyMirror/map-of-europe/master/GeoJSON/europe.geojson') as response:
        eu_nut0 = json.load(response)
    
    with urlopen('https://raw.githubusercontent.com/eurostat/Nuts2json/master/2021/4326/60M/nutsrg_2.json') as response:
        eu_nut2 = json.load(response)

    with urlopen('https://raw.githubusercontent.com/eurostat/Nuts2json/master/2021/4326/60M/nutsrg_3.json') as response:
        eu_nut3 = json.load(response)

    # selection boxes columns
    col_an = [col for col in list(table) if len(table[col].unique()) < 10 or is_numeric_dtype(table[col])]
    col_mul = [col for col in list(table) if is_numeric_dtype(table[col])]
    lis_check = [{'label': col, 'value': col} for col in col_mul if col != col_mul[0]]

    widget = st.selectbox("what is the widget you want to display:",
                          ["Ratio Analysis", "Multidimensional Analysis", "Autocorrelation Analysis", "Feature Importance Analysis", "Anomalies check", "Consistency checks"], 0)

    if widget == "Ratio Analysis":
        # ratio analysis part
        st.header("Ratio Analysis")

        st.sidebar.subheader("Ratio Area")
        ratio_num = st.sidebar.multiselect("select the ratio numerator", col_mul)
        ratio_den = st.sidebar.multiselect("select the ratio denominator", col_mul)
        
        res_ratio = pd.DataFrame(np.divide(np.nansum(table[ratio_num].values, axis = 1), np.nansum(table[ratio_den].values, axis = 1)), columns = ['R_1'])
        
        ratio_plot = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = res_ratio['R_1'].mean(),
            delta = {"reference": 2 * res_ratio['R_1'].mean() - res_ratio['R_1'].quantile(0.95)},
            domain = {'x': [0, 1], 'y': [0, 1]},
            gauge = {'axis': {'range': [res_ratio['R_1'].min(), res_ratio['R_1'].max()]},
                     'steps' : [
                         {'range': [res_ratio['R_1'].min(), res_ratio['R_1'].quantile(0.05)], 'color': "lightgray"},
                         {'range': [res_ratio['R_1'].quantile(0.95), res_ratio['R_1'].max()], 'color': "gray"}],},
            title = {'text': "Gauge plot for the variable: R_1"}))
        
        st.plotly_chart(ratio_plot, use_container_width=True)

        # map pplot + violin plot on the aggregated results
        left, right = st.columns(2)
        with left: 
            ratio_vio_sel1 = st.selectbox("multivariable index col", table.columns, 0)
        with right:
            ratio_vio_sel2 = st.selectbox("multivariable index col", ['None'] + list(table.columns), 0)
        
        res_ratio['Sel'] = table[ratio_vio_sel1].str.slice(0, 2).values
        res = {ratio_vio_sel1: res_ratio['Sel'].unique(), 'R_1': []}
        for nut_id in res[ratio_vio_sel1]:
                  res['R_1'].append(res_ratio[res_ratio['Sel'] == nut_id]['R_1'].mean())
        res = pd.DataFrame(res)

        px.set_mapbox_access_token("pk.eyJ1IjoibHVjYXVyYmFuIiwiYSI6ImNrZm5seWZnZjA5MjUydXBjeGQ5ZDBtd2UifQ.T0o-wf5Yc0iTSeq-A9Q2ww")
        map_box = px.choropleth_mapbox(res, geojson = eu_nut0, locations = res[ratio_vio_sel1], featureidkey = 'properties.ISO2',
                                       color = 'R_1', color_continuous_scale = px.colors.cyclical.IceFire,
                                       range_color = (res_ratio['R_1'].min(), res_ratio['R_1'].max()),
                                       mapbox_style = "carto-positron",
                                       zoom = 3, center = {"lat": 47.42, "lon": 15.53},
                                       opacity = 0.5,
                                       labels = {'R_1': 'R_1'})

        st.plotly_chart(map_box, use_container_width=True)
        
        uniques = list(res_ratio['Sel'].unique())
        cou_sel = st.selectbox("Choose the id of the country you want to explore", ['All ids'] + uniques, 0)
        res_ratio['Un Name'] = table['Institution Name']
        if cou_sel == 'All ids':
            if ratio_vio_sel2 == 'None':
                fig_vio = px.violin(res_ratio, y = "R_1", box = True, points = 'suspectedoutliers', title = 'Violin plot for the created ratio', hover_data = ['Un Name'])
            else:
                res_ratio['Color'] = table[ratio_vio_sel2]
                fig_vio = px.violin(res_ratio, y = "R_1", color = 'Color', box = True, points = 'suspectedoutliers', title = 'Violin plot for the created ratio', 
                                    hover_data = ['Un Name'])
        else:
            if ratio_vio_sel2 == 'None':
                fig_vio = px.violin(res_ratio[res_ratio['Sel'] == cou_sel], y = "R_1", x = 'Sel', box = True, points = 'suspectedoutliers', 
                                    title = 'Violin plot for the created ratio', hover_data = ['Un Name'])
            else:
                res_ratio['Color'] = table[ratio_vio_sel2]
                fig_vio = px.violin(res_ratio[res_ratio['Sel'] == cou_sel], y = "R_1", x = 'Sel', color = 'Color', box = True, points = 'suspectedoutliers', 
                                    title = 'Violin plot for the created ratio', hover_data = ['Un Name'])
        st.plotly_chart(fig_vio, use_container_width=True)
    
    if widget == "Multidimensional Analysis":
        # multi variable analysis part
        st.header("Multidimension Analysis")

        st.sidebar.subheader("Multivariable Area")
        multi_index = st.sidebar.selectbox("multivariable index col", table.columns, 1)
        multi_time = st.sidebar.selectbox("multivariable time col", table.columns, 3)
        multiXax_col = st.sidebar.selectbox("multivariable X axis col", col_mul, 1)
        multiYax_col = st.sidebar.selectbox("multivariable Y axis col", col_mul, 2)
        multiSlider = st.sidebar.slider("multivarible time value", int(table[multi_time].min()), int(table[multi_time].max()), int(table[multi_time].min()))

        dff = table[table[multi_time] == multiSlider]
        multi_plot = px.scatter(x = dff[multiXax_col], y = dff[multiYax_col], hover_name = dff[multi_index])
        multi_plot.update_traces(customdata = dff[multi_index])
        multi_plot.update_xaxes(title = multiXax_col)
        multi_plot.update_yaxes(title = multiYax_col)

        st.plotly_chart(multi_plot, use_container_width=True)

        # time control charts
        el_id = st.selectbox("element ID for time control chart", table[multi_index].unique(), 1)

        dff_tcc = table[table[multi_index] == el_id][[multi_time, multiXax_col, multiYax_col]]
        if len(list(dff_tcc[multi_time].unique())) < dff_tcc.shape[0]:
            res = {multi_time: [], multiXax_col: [], multiYax_col: []}
            for el in list(dff_tcc[multi_time].unique()):
                res[multi_time].append(el)
                res[multiXax_col].append(dff_tcc[dff_tcc[multi_time] == el][multiXax_col].mean())
                res[multiYax_col].append(dff_tcc[dff_tcc[multi_time] == el][multiYax_col].mean())
            dff_tcc = pd.DataFrame(data = res)
        titles = ['<b>{}</b><br>{}'.format(el_id, multiXax_col), '<b>{}</b><br>{}'.format(el_id, multiYax_col)]

        for i in range(2):
            fig_tcc = go.Figure()
            if dff_tcc.shape[0] != 0:
                x_bar = []
                for inst in table[multi_index].unique():
                    inst_data = table[table[multi_index] == inst][list(dff_tcc)[i+1]]
                    if inst_data.count() != 0:
                        x_bar.append(inst_data.mean())

                x_barbar = round(sum(x_bar)/len(x_bar), 3)

                x_LCL = x_barbar - (1.88 * (dff_tcc[list(dff_tcc)[i+1]].quantile(0.95) - dff_tcc[list(dff_tcc)[i+1]].quantile(0.05)))
                x_UCL = x_barbar + (1.88 * (dff_tcc[list(dff_tcc)[i+1]].quantile(0.95) - dff_tcc[list(dff_tcc)[i+1]].quantile(0.05)))

                x_el = [i for i in range(int(dff_tcc[multi_time].min()), int(dff_tcc[multi_time].max()) + 1)]
                fig_tcc.add_trace(go.Scatter(x = dff_tcc[multi_time], y = dff_tcc[list(dff_tcc)[i+1]], mode = 'lines+markers', name = "Value"))
                fig_tcc.add_trace(go.Scatter(x = x_el, y = [x_UCL for _ in range(len(x_el))], mode = "lines", name = "Upper Bound"))
                fig_tcc.add_trace(go.Scatter(x = x_el, y = [x_LCL for _ in range(len(x_el))], mode = "lines", name = "Lower Bound"))

                fig_tcc.update_xaxes(showgrid = False)
                fig_tcc.add_annotation(x=0, y=0.85, xanchor='left', yanchor='bottom',
                                       xref = 'paper', yref = 'paper', showarrow=False, align = 'left',
                                       bgcolor = 'rgba(255, 255, 255, 0.5)', text = titles[i])
                fig_tcc.update_layout(xaxis_title = multi_time, yaxis_title = list(dff_tcc)[i+1])
                fig_tcc.update_layout(height = 250, margin = {'l': 20, 'b': 30, 'r': 10, 't': 10})

                st.plotly_chart(fig_tcc, use_container_width=True)
    
    if widget == "Autocorrelation Analysis":
        # crossfilter analysis part
        st.header("Autocorrelation Analysis")

        st.sidebar.subheader("Autocorrelation Area")
        cross_index = st.sidebar.selectbox("autocorrelation index col", table.columns, 1)
        cross_time = st.sidebar.selectbox("autocorrelation time col", table.columns, 3)
        cross_col = st.sidebar.selectbox("autocorrelation X axis col", col_mul, 1)
        crossSlider = st.sidebar.slider("autocorrelation time value", int(table[cross_time].min()), int(table[cross_time].max()-1), int(table[cross_time].min()))

        dff_cross_dw = table[table[cross_time] == crossSlider][[cross_index, cross_col]]
        dff_cross_up = table[table[cross_time] == crossSlider + 1][[cross_index, cross_col]]
        final_df_cross = pd.merge(dff_cross_dw, dff_cross_up, how = "inner", on = cross_index)

        cross_plot = px.scatter(x = final_df_cross[cross_col + "_x"], y = final_df_cross[cross_col + "_y"], hover_name = final_df_cross[cross_index])

        cross_plot.update_xaxes(title = cross_col)
        cross_plot.update_yaxes(title = cross_col + " Next Year")

        st.plotly_chart(cross_plot, use_container_width=True)

        st.subheader("Autocorrelation")
        st.write("Autocorrelation value: " + str(round(final_df_cross[cross_col + "_x"].corr(final_df_cross[cross_col + "_y"]), 5)))

        # difference timeseries plot
        el_id_diff = st.selectbox("element ID for differences timeseries chart", table[cross_index].unique())

        dff_diff = table[table[cross_index] == el_id_diff]
        if len(list(dff_diff[cross_time].unique())) < dff_diff.shape[0]:
            res = {cross_time: [], cross_col: []}
            for el in list(dff[cross_time].unique()):
                res[cross_time].append(el); res[cross_col].append(dff_diff[dff_diff[cross_time] == el][cross_col].mean())
            dff_diff = pd.DataFrame(data = res)
        title = '<b>{}</b><br>{}'.format(el_id_diff, cross_col)

        fig_diff = go.Figure(); flag = 0
        if dff_diff.shape[0] > 1:
            x = [[i, 0] for i in range(1, dff_diff.shape[0])]
            Y = [dff_diff[cross_col].iloc[dff_diff.shape[0] - i - 1] - dff_diff[cross_col].iloc[dff_diff.shape[0] - i] for i in range(1, dff_diff.shape[0])]
            reg = LinearRegression().fit(x, Y); coeff = reg.coef_; intercept = reg.intercept_

            fig_diff.add_trace(go.Scatter(x = [str(dff_diff[cross_time].iloc[dff_diff.shape[0] - i]) + "-" + str(dff_diff[cross_time].iloc[dff_diff.shape[0] - i - 1]) for i in range(1, dff_diff.shape[0])], 
                                     y = Y, mode = 'markers', name = "Value"))
            fig_diff.add_trace(go.Scatter(x = [str(dff_diff[cross_time].iloc[dff_diff.shape[0] - i]) + "-" + str(dff_diff[cross_time].iloc[dff_diff.shape[0] - i - 1]) for i in range(1, dff_diff.shape[0])], 
                                     y = [intercept + (i * coeff[0]) for i in range(dff_diff.shape[0])], 
                                     mode = 'lines', name = "Regression"))
            fig_diff.update_xaxes(showgrid=False)
            fig_diff.add_annotation(x=0, y=0.85, xanchor='left', yanchor='bottom',
                               xref='paper', yref='paper', showarrow=False, align='left',
                               bgcolor='rgba(255, 255, 255, 0.5)', text = title)
            fig_diff.update_layout(xaxis_title = cross_time, yaxis_title = list(dff_diff)[1])
            flag = 1

        fig_diff.update_layout(height = 400)

        st.plotly_chart(fig_diff, use_container_width = True)

        st.subheader("Regression Parameters")
        if flag == 1:
            st.write("Intercept: " + str(round(intercept, 4)))
            st.write("Slope: " + str(round(coeff[0], 4)))
        else: 
            st.write("None") 
            st.write("None")
    
    if widget == "Feature Importance Analysis":
        # pareto chart with feature importance on ridge regressor
        st.sidebar.subheader("Feature Importance Area")
        feaImp_target = st.sidebar.selectbox("Feature Importance target", col_mul, 1)
        id_sel_col = st.sidebar.selectbox("ID column", table.columns, 2)

        st.header("Feature Importance Analysis")
        
        left, right = st.columns(2)
        with left: 
            fea_Imp_features = st.multiselect("Feature Importance multiselection box:", col_mul)
        with right:
            id_sel = st.selectbox("multivariable index col", ['All ids'] + list(table[id_sel_col].unique()), 0)
        
        scaler = StandardScaler()
        if id_sel == 'All ids':
            target = table[feaImp_target]
            train_nm = table[fea_Imp_features]
        else:
            target = table[table[id_sel_col] == id_sel][feaImp_target]
            train_nm = table[table[id_sel_col] == id_sel][fea_Imp_features]

        for name_col in fea_Imp_features:
            train_nm[name_col].replace({np.nan : train_nm[name_col].mean()}, inplace = True)
        train_nm = scaler.fit_transform(train_nm)

        target.replace({np.nan : 0}, inplace = True)

        Alpha = [.1, 1, 10, 100]; titles = tuple("Feature importance for alpha = " + str(alpha) for alpha in Alpha)
        Alpha = [[.1, 1], [10, 100]]

        # Create figure with secondary y-axis
        fig_tot = make_subplots(rows = 2, cols = 2, 
                                specs = [[{"secondary_y": True}, {"secondary_y": True}], 
                                         [{"secondary_y": True}, {"secondary_y": True}]], 
                                subplot_titles = titles)

        for num_row in range(2):
            for num_col in range(2):
                clf = Ridge(alpha = Alpha[num_row][num_col])
                clf.fit(train_nm, target)

                importance = clf.coef_
                for i in range(len(importance)):
                    if importance[i] < 0:
                        importance[i] *= -1
                dict_fin = {fea_Imp_features[i]: importance[i] for i in range(importance.shape[0])}
                dict_fin = {k: v for k, v in sorted(dict_fin.items(), key=lambda item: item[1], reverse = True)}
                dict_fin_per = {fea_Imp_features[i]: (importance[i] / np.sum(importance)) * 100 for i in range(importance.shape[0])}
                dict_fin_per = {k: v for k, v in sorted(dict_fin_per.items(), key=lambda item: item[1], reverse = True)}
                lis_final = []; res_par = 0
                for value in dict_fin_per.values():
                    res_par += value; lis_final.append(res_par)

                fig_tot.add_trace(
                    go.Bar(x = list(dict_fin_per.keys()), y = list(dict_fin_per.values()), 
                           marker_color = 'rgb(158,202,225)', marker_line_color = 'rgb(8,48,107)', 
                           marker_line_width = 1.5, opacity = 0.6, name = 'Value'),
                    row = num_row + 1, col = num_col + 1, secondary_y = False
                )

                fig_tot.add_trace(
                    go.Scatter(x = list(dict_fin_per.keys()), y = lis_final, line_color = 'rgb(255, 150, 0)'),
                    row = num_row + 1, col = num_col + 1, secondary_y = True
                )

                # Add figure title
                fig_tot.update_layout(
                    title_text = "Feature importances", showlegend = False
                )

                # Set x-axis title
                fig_tot.update_xaxes(title_text = "Variables")

                # Set y-axes titles
                fig_tot.update_yaxes(title_text="<b>Value</b> of importance", secondary_y=False)
                fig_tot.update_yaxes(title_text="<b>%</b> of importance", secondary_y=True)

        fig_tot.update_layout(height = 600)
        st.plotly_chart(fig_tot, use_container_width=True)
        
    if widget == "Anomalies check":
        use_col = st.sidebar.selectbox("Chosen Variable", col_mul, 0)
        var_clean = table[use_col].dropna().values
        
        # MLE normal
        mu_hat = var_clean.mean()
        sigma_hat = math.sqrt(((var_clean - var_clean.mean()) ** 2).sum() / var_clean.shape[0])

        # MLE exponential
        lambda_hat_exp = var_clean.shape[0] / var_clean.sum()

        # MLE log-normal
        mu_hat_log = (np.ma.log(var_clean)).sum() / var_clean.shape[0]
        sigma_hat_log = math.sqrt(((np.ma.log(var_clean) - mu_hat_log) ** 2).sum() / var_clean.shape[0])
        
        # MLE weibull
        a, alpha_hat, b, beta_hat = stats.exponweib.fit(var_clean, floc=0, fa=1)
        
        # computing the p-values for all the distributions
        result_norm = stats.kstest(var_clean, 'norm', (mu_hat, sigma_hat))
        result_exp = stats.kstest(var_clean, 'expon')
        result_lognorm = stats.kstest(var_clean, 'lognorm', (mu_hat_log, sigma_hat_log))
        result_weibull2 = stats.kstest(var_clean, 'dweibull', (beta_hat, alpha_hat))
        
        # visual part
        dis_fit = [[result_norm[1], result_exp[1], result_lognorm[1], result_weibull2[1]], 
                   [int(result_norm[1] > 0.05), int(result_exp[1] > 0.05), int(result_lognorm[1] > 0.05), int(result_weibull2[1] > 0.05)]]
        st.table(pd.DataFrame(dis_fit, columns = ['Normal', 'Exponential', 'Log-Norm', 'Weibul'], index = ['P-value', 'P > t']))

        ch_distr = st.selectbox("Choose the distribution you want to use for the anomalies estimation", ['Normal', 'Exponential', 'Log-Norm', 'Weibull'])
        fig_distr = go.Figure(data = [go.Histogram(x = var_clean, 
                                                   xbins = dict(start = var_clean.min(), end = var_clean.max(), size = (var_clean.max() - var_clean.min()) / 25),
                                                   autobinx = False, 
                                                   histnorm = 'probability density')])
        
        x_pos = np.linspace(var_clean.min(), var_clean.max(), 25)
        if ch_distr == 'Normal':
            fig_distr.add_trace(go.Scatter(x = x_pos, y = stats.norm(mu_hat, sigma_hat).pdf(x_pos), mode = 'lines+markers', name = "Est Distribution"))
        if ch_distr == 'Exponential':
            fig_distr.add_trace(go.Scatter(x = x_pos, y = stats.expon(lambda_hat_exp).pdf(x_pos), mode = 'lines+markers', name = "Est Distribution"))
        if ch_distr == 'Log-Norm':
            fig_distr.add_trace(go.Scatter(x = x_pos, y = stats.lognorm(mu_hat_log, sigma_hat_log).pdf(x_pos), mode = 'lines+markers', name = "Est Distribution"))
        if ch_distr == 'Weibull':
            fig_distr.add_trace(go.Scatter(x = x_pos, y = stats.dweibull(alpha_hat, beta_hat).pdf(x_pos), mode = 'lines+markers', name = "Est Distribution"))
        
        fig_distr.update_layout(title = 'Hist plot to comapre data with possible underlying distribution', xaxis_title = use_col + ' values', yaxis_title = use_col + ' PMF and ch. distr. PDF')
        st.plotly_chart(fig_distr, use_container_width=True)
         
        # outlier part
        st.markdown('In the next part there is the effective detection of the outliers contained into the data. The detection is made by the **Tukey\'s fences**. ' + 
                    'These fences are calculated by refering to the next formulas and the applied formula depend on the type of distribution chosen ' + 
                    'and the **skewness** of the data. \n If the data is **skewed** the fences are calculated in this way: ')
        st.latex(r'''[Q_{1} - (k \cdot ITQ), Q_{3} + (k \cdot ITQ)]''')
        st.markdown('If the data distribution is **not skewed** the formula changes with a correction term and becomes:')
        st.latex(r'''[Q_{1} - (k \cdot e^{-4mc} \cdot ITQ), Q_{3} + (k \cdot e^{3mc} \cdot ITQ)]''')
        st.markdown('If mc > 0 and: ')
        st.latex(r'''[Q_{1} - (k \cdot e^{-3mc} \cdot ITQ), Q_{3} + (k \cdot e^{4mc} \cdot ITQ)]''')
        st.markdown('If mc < 0. \n In these equations $Q_{1}$ represent the first quantile, while $Q_{3}$ represents the third quantile, $ITQ = Q_{3} - Q_{1}$, k is the ' + 
                    '**Tukey\'s constant** and mc is the value of the **MedCouple** function. A value is treated as an outlier if it doesen\'t fit into these intervals. ' + 
                    'In this application we make a distinction between strong and weak outlier. A strong outlier $o_{s}$ is a value that given a $t_{f}$ value for the ' + 
                    'fence\'s correction term: ')
        st.markdown('$o_{s} < Q_{1} - 2 \cdot t_{f}$ if it\'s a left outlier and $o_{s} > Q_{3} + 2 \cdot t_{f}$ if it\'s a right one')
        st.markdown('While a weak outlier $o_{w}$ is defined as: ')
        st.markdown('$o_{w} \in [Q_{1} - 2 \cdot t_{f}, Q_{1} - t_{f}]$ if it\'s a left outlier and $o_{w} \in [Q_{3} + t_{f}, Q_{3} + 2 \cdot t_{f}]$ if it\'s a right one')
        st.markdown('In the next numeric input you can insert the value of the **Tukey\'s constant** (is usually setted to 1,5), from the previous formulas we can derive ' + 
                    'that a smaller **k** will reduce the fence\'s size (you will find more outliers but less significant), while a bigger **k** will have an opposite effect,' + 
                    ' so this value must be chosen wisely.')
        
        tukey_const = st.number_input("Insert the constant for the fence interquantile value", 0.5, 7.5, 1.5)
        Q3 = table[use_col].quantile(0.75); Q1 = table[use_col].quantile(0.25); ITQ = Q3- Q1
        
        if stats.skewtest(var_clean)[1] >= 0.025:
            df_AllOut = table[(table[use_col] <= Q1 - (tukey_const * ITQ)) | (table[use_col] >= Q3 + (tukey_const * ITQ))]
            df_StLeftOut = table[table[use_col] < Q1 - (2 * tukey_const * ITQ)]
            df_WeLeftOut = table[(table[use_col] >= Q1 - (2 * tukey_const * ITQ)) & (table[use_col] <= Q1 - (tukey_const * ITQ))]
            df_WeRightOut = table[(table[use_col] >= Q3 + (tukey_const * ITQ)) & (table[use_col] <= Q3 + (2 * tukey_const * ITQ))]
            df_StRightOut = table[table[use_col] > Q3 + (2 * tukey_const * ITQ)]
            st.table(pd.DataFrame(np.array([df_StLeftOut.shape[0], df_WeLeftOut.shape[0], df_WeRightOut.shape[0], df_StRightOut.shape[0]]).reshape(1, 4),
                                  index = ['Number'], columns = ['Strong left outliers', 'Weak left outliers', 'Weak right outliers', 'Strong right outliers']))
        else:
            # calculating the medcouple function for the tukey fence
            if var_clean.shape[0] > 5000:
                MC = np.array([medcouple(var_clean[np.random.choice(var_clean.shape[0], 5000)]) for _ in range(50)]).mean()
            else:
                MC = medcouple(var_clean)
            
            # calculating the tukey fence
            if MC > 0:
                df_AllOut = table[(table[use_col] <= Q1 - (tukey_const * math.exp(-4 * MC) * ITQ)) | (table[use_col] >= Q3 + (tukey_const * math.exp(3 * MC) * ITQ))]
                df_StLeftOut = table[table[use_col] < Q1 - (2 * tukey_const * math.exp(-4 * MC) * ITQ)]
                df_WeLeftOut = table[(table[use_col] >= Q1 - (2 * tukey_const * math.exp(-4 * MC) * ITQ)) & (table[use_col] <= Q1 - (tukey_const * math.exp(-4 * MC) * ITQ))]
                df_WeRightOut = table[(table[use_col] >= Q3 + (tukey_const * math.exp(3 * MC) * ITQ)) & (table[use_col] <= Q3 + (2 * tukey_const * math.exp(3 * MC) * ITQ))]
                df_StRightOut = table[table[use_col] > Q3 + (2 * tukey_const * math.exp(3 * MC) * ITQ)]
                st.table(pd.DataFrame(np.array([df_StLeftOut.shape[0], df_WeLeftOut.shape[0], df_WeRightOut.shape[0], df_StRightOut.shape[0]]).reshape(1, 4),
                                      index = ['Number'], columns = ['Strong left outliers', 'Weak left outliers', 'Weak right outliers', 'Strong right outliers']))
            else:
                df_AllOut = table[(table[use_col] <= Q1 - (tukey_const * math.exp(-3 * MC) * ITQ)) | (table[use_col] >= Q3 + (tukey_const * math.exp(4 * MC) * ITQ))]
                df_StLeftOut = table[table[use_col] < Q1 - (2 * tukey_const * math.exp(-3 * MC) * ITQ)]
                df_WeLeftOut = table[(table[use_col] >= Q1 - (2 * tukey_const * math.exp(-3 * MC) * ITQ)) & (table[use_col] <= Q1 - (tukey_const * math.exp(-3 * MC) * ITQ))]
                df_WeRightOut = table[(table[use_col] >= Q3 + (tukey_const * math.exp(4 * MC) * ITQ)) & (table[use_col] <= Q3 + (2 * tukey_const * math.exp(4 * MC) * ITQ))]
                df_StRightOut = table[table[use_col] > Q3 + (2 * tukey_const * math.exp(4 * MC) * ITQ)]
                st.table(pd.DataFrame(np.array([df_StLeftOut.shape[0], df_WeLeftOut.shape[0], df_WeRightOut.shape[0], df_StRightOut.shape[0]]).reshape(1, 4),
                                      index = ['Number'], columns = ['Strong left outliers', 'Weak left outliers', 'Weak right outliers', 'Strong right outliers']))
        
        # a more specific view of the ouliers by country or generic id and type
        left, right = st.columns(2)
        with left: 
            out_id_col = st.selectbox("Outlier index col", table.columns, 0)
        with right:
            out_type = st.selectbox("Type of outliers you want to investigate", ['All', 'Strong left outliers', 'Weak left outliers', 'Weak right outliers', 'Strong right outliers'], 0)
        
        if out_type == 'All':
            df_AllOut['Sel'] = df_AllOut[out_id_col].str.slice(0, 2).values
            res = {out_id_col: df_AllOut['Sel'].unique(), 'Num. Out.': []}
            for nut_id in res[out_id_col]:
                  res['Num. Out.'].append(df_AllOut[df_AllOut['Sel'] == nut_id].shape[0])
            res = pd.DataFrame(res)

            px.set_mapbox_access_token("pk.eyJ1IjoibHVjYXVyYmFuIiwiYSI6ImNrZm5seWZnZjA5MjUydXBjeGQ5ZDBtd2UifQ.T0o-wf5Yc0iTSeq-A9Q2ww")
            map_box = px.choropleth_mapbox(res, geojson = eu_nut0, locations = res[out_id_col], featureidkey = 'properties.ISO2',
                                           color = 'Num. Out.', color_continuous_scale = px.colors.cyclical.IceFire,
                                           range_color = (res['Num. Out.'].min(), res['Num. Out.'].max()),
                                           mapbox_style = "carto-positron",
                                           zoom = 3, center = {"lat": 47.42, "lon": 15.53},
                                           opacity = 0.5,
                                           labels = {'Num. Out.': 'Num. Out.'})
            st.plotly_chart(map_box, use_container_width=True)
         
        if out_type == 'Strong left outliers':
            df_StLeftOut['Sel'] = df_StLeftOut[out_id_col].str.slice(0, 2).values
            res = {out_id_col: df_StLeftOut['Sel'].unique(), 'Num. Out.': []}
            for nut_id in res[out_id_col]:
                  res['Num. Out.'].append(df_StLeftOut[df_StLeftOut['Sel'] == nut_id].shape[0])
            res = pd.DataFrame(res)

            px.set_mapbox_access_token("pk.eyJ1IjoibHVjYXVyYmFuIiwiYSI6ImNrZm5seWZnZjA5MjUydXBjeGQ5ZDBtd2UifQ.T0o-wf5Yc0iTSeq-A9Q2ww")
            map_box = px.choropleth_mapbox(res, geojson = eu_nut0, locations = res[out_id_col], featureidkey = 'properties.ISO2',
                                           color = 'Num. Out.', color_continuous_scale = px.colors.cyclical.IceFire,
                                           range_color = (res['Num. Out.'].min(), res['Num. Out.'].max()),
                                           mapbox_style = "carto-positron",
                                           zoom = 3, center = {"lat": 47.42, "lon": 15.53},
                                           opacity = 0.5,
                                           labels = {'Num. Out.': 'Num. Out.'})
            st.plotly_chart(map_box, use_container_width=True)
            
        if out_type == 'Weak left outliers':
            df_WeLeftOut['Sel'] = df_WeLeftOut[out_id_col].str.slice(0, 2).values
            res = {out_id_col: df_WeLeftOut['Sel'].unique(), 'Num. Out.': []}
            for nut_id in res[out_id_col]:
                  res['Num. Out.'].append(df_WeLeftOut[df_WeLeftOut['Sel'] == nut_id].shape[0])
            res = pd.DataFrame(res)

            px.set_mapbox_access_token("pk.eyJ1IjoibHVjYXVyYmFuIiwiYSI6ImNrZm5seWZnZjA5MjUydXBjeGQ5ZDBtd2UifQ.T0o-wf5Yc0iTSeq-A9Q2ww")
            map_box = px.choropleth_mapbox(res, geojson = eu_nut0, locations = res[out_id_col], featureidkey = 'properties.ISO2',
                                           color = 'Num. Out.', color_continuous_scale = px.colors.cyclical.IceFire,
                                           range_color = (res['Num. Out.'].min(), res['Num. Out.'].max()),
                                           mapbox_style = "carto-positron",
                                           zoom = 3, center = {"lat": 47.42, "lon": 15.53},
                                           opacity = 0.5,
                                           labels = {'Num. Out.': 'Num. Out.'})
            st.plotly_chart(map_box, use_container_width=True)
            
        if out_type == 'Weak right outliers':
            df_WeRightOut['Sel'] = df_WeRightOut[out_id_col].str.slice(0, 2).values
            res = {out_id_col: df_WeRightOut['Sel'].unique(), 'Num. Out.': []}
            for nut_id in res[out_id_col]:
                  res['Num. Out.'].append(df_WeRightOut[df_WeRightOut['Sel'] == nut_id].shape[0])
            res = pd.DataFrame(res)

            px.set_mapbox_access_token("pk.eyJ1IjoibHVjYXVyYmFuIiwiYSI6ImNrZm5seWZnZjA5MjUydXBjeGQ5ZDBtd2UifQ.T0o-wf5Yc0iTSeq-A9Q2ww")
            map_box = px.choropleth_mapbox(res, geojson = eu_nut0, locations = res[out_id_col], featureidkey = 'properties.ISO2',
                                           color = 'Num. Out.', color_continuous_scale = px.colors.cyclical.IceFire,
                                           range_color = (res['Num. Out.'].min(), res['Num. Out.'].max()),
                                           mapbox_style = "carto-positron",
                                           zoom = 3, center = {"lat": 47.42, "lon": 15.53},
                                           opacity = 0.5,
                                           labels = {'Num. Out.': 'Num. Out.'})
            st.plotly_chart(map_box, use_container_width=True)
            
        if out_type == 'Strong right outliers':
            df_StRightOut['Sel'] = df_StRightOut[out_id_col].str.slice(0, 2).values
            res = {out_id_col: df_StRightOut['Sel'].unique(), 'Num. Out.': []}
            for nut_id in res[out_id_col]:
                  res['Num. Out.'].append(df_StRightOut[df_StRightOut['Sel'] == nut_id].shape[0])
            res = pd.DataFrame(res)

            px.set_mapbox_access_token("pk.eyJ1IjoibHVjYXVyYmFuIiwiYSI6ImNrZm5seWZnZjA5MjUydXBjeGQ5ZDBtd2UifQ.T0o-wf5Yc0iTSeq-A9Q2ww")
            map_box = px.choropleth_mapbox(res, geojson = eu_nut0, locations = res[out_id_col], featureidkey = 'properties.ISO2',
                                           color = 'Num. Out.', color_continuous_scale = px.colors.cyclical.IceFire,
                                           range_color = (res['Num. Out.'].min(), res['Num. Out.'].max()),
                                           mapbox_style = "carto-positron",
                                           zoom = 3, center = {"lat": 47.42, "lon": 15.53},
                                           opacity = 0.5,
                                           labels = {'Num. Out.': 'Num. Out.'})
            st.plotly_chart(map_box, use_container_width=True)
        
        out_cou = st.selectbox("Choose the specific value for the id", ['All ids'] + list(res[out_id_col]), 0)
         
        if out_cou == 'All ids': 
            st.write(df_AllOut)
        else:
            st.write(df_AllOut) 
        
    if widget == "Consistency checks":
        methodology = st.sidebar.selectbox("Choose the type of methodology you want to apply", ['Multiannual methodology', 'Ratio methodology'], 0)
        if methodology == 'Ratio methodology':
            con_checks_id_col = st.sidebar.selectbox("Index col", table.columns, 0)
            country_sel_col = st.sidebar.selectbox("Country selection column", ['-'] + list(table.columns), 0)
            cat_sel_col = st.sidebar.selectbox("Category selection column", ['-'] + list(table.columns), 0)
            flag_issue_quantile = st.sidebar.number_input("Insert the quantile that will issue the flag (S2 and S3)", 0.0, 10.0, 5.0, 0.1)
            prob_cases_per = st.sidebar.number_input("Insert the percentage for the problematic cases", 0.0, 100.0, 20.0)
            p_value_trend_per = st.sidebar.number_input("Insert the p-value percentage for the trend estimation", 5.0, 50.0, 10.0)

            con_checks_feature = st.selectbox("Variables chosen for the consistency checks:", col_mul)
            flag_radio = st.radio("Do you want to use the flags:", ('Yes', 'No'))
            if flag_radio == 'Yes':
                left1, right1 = st.columns(2)
                with left1:
                    flags_col = st.selectbox("Select the specific flag variable for the checks", table.columns)
                with right1:
                    notes_col = st.selectbox("Select the specific flag notes variable for the checks", ['-'] + list(table.columns))
            
            table['Class trend'] = 0
            for id_inst in table[con_checks_id_col].unique():
                # trend classification
                inst = table[table[con_checks_id_col] == id_inst][con_checks_feature].values[::-1]
                geo_mean_vec = np.delete(inst, np.where((inst == 0) | (np.isnan(inst))))
                if geo_mean_vec.shape[0] > 3:
                    mann_kend_res = mk.original_test(geo_mean_vec)
                    trend, p, tau = mann_kend_res.trend, mann_kend_res.p, mann_kend_res.Tau
                    if trend == 'increasing':
                        table.loc[table[table[con_checks_id_col] == id_inst].index, 'Class trend'] = 5
                    if trend == 'decreasing':
                        table.loc[table[table[con_checks_id_col] == id_inst].index, 'Class trend'] = 1
                    if trend == 'no trend':
                        if p <= p_value_trend_per/100 and tau >= 0:
                            table.loc[table[table[con_checks_id_col] == id_inst].index, 'Class trend'] = 4
                        if p <= p_value_trend_per/100 and tau < 0:
                            table.loc[table[table[con_checks_id_col] == id_inst].index, 'Class trend'] = 2
                        if p > p_value_trend_per/100:
                            table.loc[table[table[con_checks_id_col] == id_inst].index, 'Class trend'] = 3
                
            dict_flags = dict(); countries = list(table[country_sel_col].unique())
            if cat_sel_col != '-':
                categories = list(table[cat_sel_col].unique())
                dict_flags[con_checks_feature] = dict()
                for cc in countries:
                    country_table = table[table[country_sel_col] == cc][[con_checks_id_col, con_checks_feature]]
                    inst_lower = set(country_table[country_table[con_checks_feature] <= country_table[con_checks_feature].quantile(flag_issue_quantile/100)]['ETER ID'].values)
                    inst_upper = set(country_table[country_table[con_checks_feature] >= country_table[con_checks_feature].quantile(1 - (flag_issue_quantile/100))]['ETER ID'].values)
                    dict_flags[con_checks_feature][cc] = inst_lower.union(inst_upper)
                for cat in categories:
                    cat_table = table[table[cat_sel_col] == cat][[con_checks_id_col, con_checks_feature]]
                    inst_lower = set(cat_table[cat_table[con_checks_feature] <= cat_table[con_checks_feature].quantile(flag_issue_quantile/100)]['ETER ID'].values)
                    inst_upper = set(cat_table[cat_table[con_checks_feature] >= cat_table[con_checks_feature].quantile(1 - (flag_issue_quantile/100))]['ETER ID'].values)
                    dict_flags[con_checks_feature][cat] = inst_lower.union(inst_upper)

                dict_check_flags = {}; set_app = set()
                for cc in countries:
                    set_app = set_app.union(dict_flags[con_checks_feature][cc])
                for cat in categories:
                    set_app = set_app.union(dict_flags[con_checks_feature][cat])
                dict_check_flags[con_checks_feature] = set_app

                table['Prob inst ' + con_checks_feature] = 0
                table.loc[table[table[con_checks_id_col].isin(dict_check_flags[con_checks_feature])].index, 'Prob inst ' + con_checks_feature] = 1

                # table reporting the cases by countries
                DV_fin_res = np.zeros((len(categories), len(countries)), dtype = int)
                for j in range(len(countries)):
                    for el in dict_flags[con_checks_feature][countries[j]]:
                        DV_fin_res[categories.index(table[table[con_checks_id_col] == el][cat_sel_col].unique()[0]), j] += 1
                for j in range(len(categories)):
                    for el in dict_flags[con_checks_feature][categories[j]]:
                        if el not in dict_flags[con_checks_feature][countries[countries.index(el[:2])]]:
                            DV_fin_res[j, countries.index(el[:2])] += 1

                DV_fin_res = np.append(DV_fin_res, np.sum(DV_fin_res, axis = 1).reshape((len(categories), 1)), axis = 1)
                DV_fin_res = np.append(DV_fin_res, np.sum(DV_fin_res, axis = 0).reshape(1, len(countries) + 1), axis = 0)
                list_fin_res = DV_fin_res.tolist(); list_prob_cases = []
                for row in range(len(list_fin_res)):
                    for i in range(len(list_fin_res[row])):
                        if row != len(list_fin_res)-1 and i != len(list_fin_res[row])-1:
                            den = len(table[(table[country_sel_col] == countries[i]) & (table[cat_sel_col] == categories[row])][con_checks_id_col].unique())
                        if row == len(list_fin_res)-1 and i != len(list_fin_res[row])-1:
                            den = len(table[table[country_sel_col] == countries[i]][con_checks_id_col].unique())
                        if row != len(list_fin_res)-1 and i == len(list_fin_res[row])-1:
                            den = len(table[table[cat_sel_col] == categories[row]][con_checks_id_col].unique())
                        if row == len(list_fin_res)-1 and i == len(list_fin_res[row])-1:
                            den = table.shape[0]
                        num = list_fin_res[row][i]
                        if den != 0:
                            num_app = round(100 * num/den, 2); list_fin_res[row][i] = str(list_fin_res[row][i]) + '\n(' + str(num_app) + '%)'
                        else:
                            num_app = 0; list_fin_res[row][i] = '0\n(0%)'
                        if i != len(list_fin_res[row])-1 and num_app >= prob_cases_per:
                            if row != len(list_fin_res)-1:
                                list_prob_cases.append([con_checks_feature, countries[i], categories[int(row % len(categories))], str(num_app) + '%', str(num) + ' / ' + str(den)])
                            else:
                                list_prob_cases.append(['Total', countries[i], 'All categories', str(num_app) + '%', str(num) + ' / ' + str(den)])

                flag_notes_on = False
                if flag_radio == 'Yes':
                    if table[flags_col].dtypes == 'O':
                        if notes_col == '-':
                            ones = set(table[(-pd.isna(table[flags_col])) & (table[flags_col] != 'p')][con_checks_id_col].values)
                        else:
                            flag_notes_on = True
                            ones = set(table[(-pd.isna(table[flags_col])) & (table[flags_col] != 'p') & (pd.isna(table[notes_col]))][con_checks_id_col].values).union(set(table[table[flags_col] == 'p'][con_checks_id_col].values))
                            twos = set(table[(-pd.isna(table[flags_col])) & (table[flags_col] != 'p') & (-pd.isna(table[notes_col]))][con_checks_id_col].values)
                            ones = ones - (ones & twos)
                    else:
                        ones = set(table[table[flags_col] == 1][con_checks_id_col].values); twos = set(table[table[flags_col] == 2][con_checks_id_col].values)
                    if flag_notes_on:
                        summ_table = pd.DataFrame([[str(len(twos.intersection(dict_check_flags))) + ' over ' + str(len(twos)), str(round((100 * len(twos.intersection(dict_check_flags))) / len(twos), 2)) + '%'], 
                                                   [str(len(dict_check_flags)) + ' / ' + str(len(ones.union(twos))), str(round(100 * (len(dict_check_flags) / len(ones.union(twos))), 2)) + '%'], 
                                                   [str(len(dict_check_flags.difference(ones.union(twos)))), str(round((100 * len(dict_check_flags.difference(ones.union(twos)))) / len(dict_check_flags), 2)) + '%']], 
                                                   columns = ['Absolute Values', 'In percentage'], 
                                                   index = ['Accuracy respect the confirmed cases', '#application cases vs. #standard cases', 'Number of not flagged cases'])
                    else:
                        summ_table = pd.DataFrame([[str(len(ones.intersection(dict_check_flags))) + ' over ' + str(len(ones)), str(round((100 * len(ones.intersection(dict_check_flags))) / len(ones), 2)) + '%'], 
                                                   [str(len(dict_check_flags)) + ' / ' + str(len(ones)), str(round(100 * (len(dict_check_flags) / len(ones)), 2)) + '%'], 
                                                   [str(len(dict_check_flags.difference(ones))), str(round((100 * len(dict_check_flags.difference(ones))) / len(dict_check_flags), 2)) + '%']], 
                                                   columns = ['Absolute Values', 'In percentage'], 
                                                   index = ['Accuracy respect the confirmed cases', '#application cases vs. #standard cases', 'Number of not flagged cases'])
                    st.table(summ_table)
                
                table_fin_res = pd.DataFrame(list_fin_res, index = [con_checks_feature + ' (' + str(cat) + ')' for cat in categories] + ['Total'], columns = countries + ['Total'])
                st.table(table_fin_res)
                st.table(pd.DataFrame(list_prob_cases, columns = ['Variable', 'Country', 'Category', '% Value', 'Absolute values']))

                if len(list(table['Class trend'].unique())) != 1:
                    dict_trend = {'Strong decrease': [], 'Weak decrease': [], 'Undetermined trend': [], 'Weak increase': [], 'Strong increase': []}; set_trend = set()
                    for inst in dict_check_flags[con_checks_feature]:
                        class_tr = int(table[table[con_checks_id_col] == inst]['Class trend'].unique()[0])
                        if class_tr != 0:
                            dict_trend[list(dict_trend.keys())[class_tr-1]].append(inst)
                            if class_tr == 1 or class_tr == 3 or class_tr == 5:
                                set_trend.add(inst)
                    trend_table = pd.DataFrame([len(v) for v in dict_trend.values()], index = dict_trend.keys(), columns = ['Number of institutions'])
                    
                    st.table(trend_table)
                    if flag_radio == 'Yes':
                        ones = set(table[table[flags_col] == 1][con_checks_id_col].values); twos = set(table[table[flags_col] == 2][con_checks_id_col].values)
                        st.table(pd.DataFrame([[str(len(twos.intersection(set_trend))) + ' over ' + str(len(twos)), str(round((100 * len(twos.intersection(set_trend))) / len(twos), 2)) + '%'], 
                                               [str(len(set_trend)) + ' / ' + str(len(ones.union(twos))), str(round(100 * (len(set_trend) / len(ones.union(twos))), 2)) + '%'], 
                                               [str(len(set_trend.difference(ones.union(twos)))), '0%']], 
                                               columns = ['Absolute Values', 'In percentage'], 
                                               index = ['Accuracy respect the confirmed cases', '#application cases vs. #standard cases', 'Number of not flagged cases']))

                    trend_type = st.selectbox('Choose the institution trend type you want to vizualize', list(dict_trend.keys()), 0)
                    trend_inst = st.selectbox('Choose the institution you want to vizualize', dict_trend[trend_type])
                    st.plotly_chart(px.line(table[table[con_checks_id_col] == trend_inst][[con_checks_feature, 'Reference year']], 
                                            x = 'Reference year', y = con_checks_feature), use_container_width=True)
                
                cols_pr_inst = st.multiselect('Choose the variables', col_mul); dict_pr_inst = {}
                for col in cols_pr_inst:
                    dict_flags[col] = dict()
                    for cc in countries:
                        country_table = table[table[country_sel_col] == cc][[con_checks_id_col, col]]
                        inst_lower = set(country_table[country_table[col] <= country_table[col].quantile(0.05)][con_checks_id_col].values)
                        inst_upper = set(country_table[country_table[col] >= country_table[col].quantile(1 - (0.05))][con_checks_id_col].values)
                        dict_flags[col][cc] = inst_lower.union(inst_upper)
                    for cat in categories:
                        cat_table = table[table[cat_sel_col] == cat][[con_checks_id_col, col]]
                        inst_lower = set(cat_table[cat_table[col] <= cat_table[col].quantile(0.05)][con_checks_id_col].values)
                        inst_upper = set(cat_table[cat_table[col] >= cat_table[col].quantile(1 - (0.05))][con_checks_id_col].values)
                        dict_flags[col][cat] = inst_lower.union(inst_upper)

                    dict_check_flags = {}; set_app = set()
                    for cc in countries:
                        set_app = set_app.union(dict_flags[col][cc])
                    for cat in categories:
                        set_app = set_app.union(dict_flags[col][cat])
                    dict_check_flags[col] = set_app

                    for inst in dict_check_flags[col]:
                        if inst not in dict_pr_inst.keys():
                            dict_pr_inst[inst] = [col]
                        else:
                            dict_pr_inst[inst].append(col)

                dict_pr_inst = dict(sorted(dict_pr_inst.items(), key = lambda item: len(item[1]), reverse = True))
                dict_pr_inst = {k: [len(v), ' '.join(v)] for k, v in dict_pr_inst.items()}
                st.table(pd.DataFrame(dict_pr_inst.values(), index = dict_pr_inst.keys(), columns = ['# of problems', 'Probematic variables']).head(25))
                
                st.write('If you want to download the result file with all the issued flags you have first to choose at least the time column and then to clik on the following button:')
                left1, right1 = st.columns(2)
                with left1:
                    time_col = st.selectbox("Select the variable from wich you want to extract the time values:", table.columns)
                with right1:
                    descr_col = st.multiselect("Select the desciptive columns you want to add to the result dataset:", table.columns)

                t_col = [str(el) for el in sorted(table[time_col].unique())]; list_fin = []
                if flag_radio == 'Yes':
                    df_cols = [con_checks_id_col] + descr_col + t_col + ['Variable', 'Trend', 'Meta flag', 'Application Flag']
                else:
                    df_cols = [con_checks_id_col] + descr_col + t_col + ['Variable', 'Trend', 'Application Flag']
                for inst in sorted(list(table[con_checks_id_col].unique())):
                    df_inst = table[table[con_checks_id_col] == inst]
                    list_el = [inst]
                    for col in descr_col:
                        list_el.append(df_inst[col].unique()[0])
                    for t in t_col:
                        if df_inst[df_inst[time_col] == int(t)].shape[0] != 0:
                            list_el.append(df_inst[df_inst[time_col] == int(t)][con_checks_feature].values[0])
                        else:
                            list_el.append(np.nan)
                    list_el.append(con_checks_feature)
                    list_el.append(df_inst['Class trend'].unique()[0])
                    if flag_radio == 'Yes':
                        list_el.append(df_inst[flags_col].unique()[0])
                    list_el.append(df_inst['Prob inst ' + con_checks_feature].unique()[0])
                    list_fin.append(list_el)
                table_download = pd.DataFrame(list_fin, columns = df_cols)
                st.download_button(label = "Download data with lables", data = table_download.to_csv(index = None, sep = ';').encode('utf-8'), file_name = 'result.csv', mime = 'text/csv')  
            else:
                st.warning('you have to choose a value for the field "Category selection column".')
        else:
            con_checks_id_col = st.sidebar.selectbox("Index col", table.columns, 0)
            country_sel_col = st.sidebar.selectbox("Country selection column", ['-'] + list(table.columns), 0)
            cat_sel_col = st.sidebar.selectbox("Category selection column", ['-'] + list(table.columns), 0)
            retain_quantile = st.sidebar.number_input("Insert the quantile you want to exclude from the calculations (S1)", 1.0, 10.0, 2.0, 0.1)
            flag_issue_quantile = st.sidebar.number_input("Insert the quantile that will issue the flag (S2 and S3)", 35.0, 100.0, 95.0, 0.1)
            prob_cases_per = st.sidebar.number_input("Insert the percentage for the problematic cases", 0.0, 100.0, 20.0)
            p_value_trend_per = st.sidebar.number_input("Insert the p-value percentage for the trend estimation", 5.0, 50.0, 10.0)

            con_checks_features = st.selectbox("Variables chosen for the consistency checks:", col_mul)
            flag_radio = st.radio("Do you want to use the flags:", ('Yes', 'No'))
            if flag_radio == 'Yes':
                left1, right1 = st.columns(2)
                with left1:
                    flags_col = st.selectbox("Select the specific flag variable for the checks", table.columns)
                with right1:
                    notes_col = st.selectbox("Select the specific flag notes variable for the checks", ['-'] + list(table.columns))
                
            res_ind = dict(); table['Class trend'] = 0
            for id_inst in table[con_checks_id_col].unique():
                # calculations of the geometric mean
                inst = table[table[con_checks_id_col] == id_inst][con_checks_features].values[::-1]
                geo_mean_vec = np.delete(inst, np.where((inst == 0) | (np.isnan(inst))))
                if geo_mean_vec.shape[0] != 0:
                    res_ind[id_inst] = math.pow(math.fabs(np.prod(geo_mean_vec)), 1/geo_mean_vec.shape[0])
                else:
                    res_ind[id_inst] = np.nan
                    
                # trend classification
                if geo_mean_vec.shape[0] > 3:
                    mann_kend_res = mk.original_test(geo_mean_vec)
                    trend, p, tau = mann_kend_res.trend, mann_kend_res.p, mann_kend_res.Tau
                    if trend == 'increasing':
                        table.loc[table[table[con_checks_id_col] == id_inst].index, 'Class trend'] = 5
                    if trend == 'decreasing':
                        table.loc[table[table[con_checks_id_col] == id_inst].index, 'Class trend'] = 1
                    if trend == 'no trend':
                        if p <= p_value_trend_per/100 and tau >= 0:
                            table.loc[table[table[con_checks_id_col] == id_inst].index, 'Class trend'] = 4
                        if p <= p_value_trend_per/100 and tau < 0:
                            table.loc[table[table[con_checks_id_col] == id_inst].index, 'Class trend'] = 2
                        if p > p_value_trend_per/100:
                            table.loc[table[table[con_checks_id_col] == id_inst].index, 'Class trend'] = 3
            
            indices = pd.DataFrame(res_ind.values(), index = res_ind.keys(), columns = [con_checks_features])
            indices.drop(index = set(indices[(pd.isna(indices[con_checks_features])) | (indices[con_checks_features] <= indices.quantile(retain_quantile/100).values[0])].index), axis = 0, inplace = True)

            res = dict(); list_prob_cases = []
            # does the calculation with the delta+ and delta-minus for the multiannual checks and stores it into a dictionary 
            for id_inst in indices.index.values:
                inst = table[(table[con_checks_id_col] == id_inst) & (-pd.isna(table[con_checks_features]))][con_checks_features].values
                num_row = len(inst); delta_pos = list(); delta_neg = list()
                for i in range(1, num_row):
                    if inst[num_row - i - 1] - inst[num_row - i] < 0:
                        delta_neg.append(round(inst[num_row - i - 1] - inst[num_row - i], 2))
                    else:
                        delta_pos.append(round(inst[num_row - i - 1] - inst[num_row - i], 2))
                res[id_inst] = [delta_pos, delta_neg]

            DV = dict() # the dictionary in wich we'll store all the DV and further the DM values for the variability from years
            for key, value in res.items():
                res_par = 0
                if len(value[0]) != 0 and len(value[1]) != 0:
                    res_par = sum(value[0]) * sum(value[1])
                DV[key] = round(math.fabs(res_par)/indices[con_checks_features][key] ** 1.5, 3)

                DV_df = pd.DataFrame(DV.values(), index = DV.keys(), columns = [con_checks_features])
                dict_check_flags = set(DV_df[DV_df[con_checks_features] >= DV_df[con_checks_features].quantile(flag_issue_quantile/100)].index)

            for el in table[country_sel_col].unique():
                if len(el) > 2:
                    table['New Country Code'] = table[country_sel_col].str[:2]
                    country_sel_col = 'New Country Code'
                    break
            list_countries = list(table[country_sel_col].unique())
             
            if cat_sel_col == '-':
                DV_fin_res = np.zeros((1, len(list_countries)), dtype = int)
                for flag in dict_check_flags:
                    DV_fin_res[0, list_countries.index(flag[:2])] += 1
            else:
                list_un_cat = list(table[cat_sel_col].unique())
                DV_fin_res = np.zeros((len(list_un_cat), len(list_countries)), dtype = int)
                for flag in dict_check_flags:
                    DV_fin_res[list_un_cat.index(table[table[con_checks_id_col] == flag][cat_sel_col].unique()[0]), list_countries.index(flag[:2])] += 1

            table['Prob inst ' + con_checks_features] = 0
            table.loc[table[table[con_checks_id_col].isin(dict_check_flags)].index, 'Prob inst ' + con_checks_features] = 1
                        
            if cat_sel_col == '-':
                DV_fin_res = np.append(DV_fin_res, np.array([np.sum(DV_fin_res, axis = 0)]), axis = 0)
                DV_fin_res = np.append(DV_fin_res, DV_fin_res, axis = 1)
                list_fin_res = DV_fin_res.tolist()
                for row in range(len(list_fin_res)):
                    for i in range(len(list_fin_res[row])):
                        if i != len(list_fin_res[row])-1:
                            den = len(table[table[country_sel_col] == list_countries[i]][con_checks_id_col].unique())
                        else:
                            den = len(table[con_checks_id_col].unique())
                        num = list_fin_res[row][i]
                        if den != 0:
                            num_app = round(100 * num/den, 2); list_fin_res[row][i] = str(list_fin_res[row][i]) + '\n(' + str(num_app) + '%)'
                        else:
                            num_app = 0; list_fin_res[row][i] = '0\n(0%)'
                        if i != len(list_fin_res[row])-1 and num_app >= prob_cases_per:
                            if row != len(list_fin_res)-1:
                                list_prob_cases.append([con_checks_features, list_countries[i], str(num_app) + '%', str(num) + ' / ' + str(den)])
                            else:
                                list_prob_cases.append(['Total', list_countries[i], str(num_app) + '%', str(num) + ' / ' + str(den)])
                table_fin_res = pd.DataFrame(list_fin_res, index = [con_checks_features, 'Total'], columns = list_countries + ['Total'])
            else:
                DV_fin_res = np.append(DV_fin_res, np.sum(DV_fin_res, axis = 1).reshape((len(list_un_cat), 1)), axis = 1)
                DV_fin_res = np.append(DV_fin_res, np.sum(DV_fin_res, axis = 0).reshape(1, len(list_countries)+1), axis = 0)
                list_fin_res = DV_fin_res.tolist()
                for row in range(len(list_fin_res)):
                    for i in range(len(list_fin_res[row])):
                        if row != len(list_fin_res)-1 and i != len(list_fin_res[row])-1:
                            den = len(table[(table[country_sel_col] == list_countries[i]) & (table[cat_sel_col] == list_un_cat[row])][con_checks_id_col].unique())
                        if row == len(list_fin_res)-1 and i != len(list_fin_res[row])-1:
                            den = len(table[table[country_sel_col] == list_countries[i]][con_checks_id_col].unique())
                        if row != len(list_fin_res)-1 and i == len(list_fin_res[row])-1:
                            den = len(table[table[cat_sel_col] == list_un_cat[row]][con_checks_id_col].unique())
                        if row == len(list_fin_res)-1 and i == len(list_fin_res[row])-1:
                            den = len(table[con_checks_id_col].unique())
                        num = list_fin_res[row][i]
                        if den != 0:
                            num_app = round(100 * num/den, 2); list_fin_res[row][i] = str(list_fin_res[row][i]) + '\n(' + str(num_app) + '%)'
                        else:
                            num_app = 0; list_fin_res[row][i] = '0\n(0%)'
                        if i != len(list_fin_res[row])-1 and num_app >= prob_cases_per:
                            if row != len(list_fin_res)-1:
                                list_prob_cases.append([con_checks_features, list_countries[i], list_un_cat[int(row % len(list_un_cat))], str(num_app) + '%', str(num) + ' / ' + str(den)])
                            else:
                                list_prob_cases.append(['Total', list_countries[i], 'All categories', str(num_app) + '%', str(num) + ' / ' + str(den)])
                table_fin_res = pd.DataFrame(list_fin_res, index = [con_checks_features + ' (' + str(cat) + ')' for cat in list_un_cat] + ['Total'], columns = list_countries + ['Total'])

            flag_notes_on = False
            if flag_radio == 'Yes':
                if table[flags_col].dtypes == 'O':
                    if notes_col == '-':
                        ones = set(table[(-pd.isna(table[flags_col])) & (table[flags_col] != 'p')][con_checks_id_col].values)
                    else:
                        flag_notes_on = True
                        ones = set(table[(-pd.isna(table[flags_col])) & (table[flags_col] != 'p') & (pd.isna(table[notes_col]))][con_checks_id_col].values).union(set(table[table[flags_col] == 'p'][con_checks_id_col].values))
                        twos = set(table[(-pd.isna(table[flags_col])) & (table[flags_col] != 'p') & (-pd.isna(table[notes_col]))][con_checks_id_col].values)
                        ones = ones - (ones & twos)
                else:
                    ones = set(table[table[flags_col] == 1][con_checks_id_col].values); twos = set(table[table[flags_col] == 2][con_checks_id_col].values)
                if flag_notes_on:
                    summ_table = pd.DataFrame([[str(len(twos.intersection(dict_check_flags))) + ' over ' + str(len(twos)), str(round((100 * len(twos.intersection(dict_check_flags))) / len(twos), 2)) + '%'], 
                                               [str(len(dict_check_flags)) + ' / ' + str(len(ones.union(twos))), str(round(100 * (len(dict_check_flags) / len(ones.union(twos))), 2)) + '%'], 
                                               [str(len(dict_check_flags.difference(ones.union(twos)))), str(round((100 * len(dict_check_flags.difference(ones.union(twos)))) / len(dict_check_flags), 2)) + '%']], 
                                               columns = ['Absolute Values', 'In percentage'], 
                                               index = ['Accuracy respect the confirmed cases', '#application cases vs. #standard cases', 'Number of not flagged cases'])
                else:
                    summ_table = pd.DataFrame([[str(len(ones.intersection(dict_check_flags))) + ' over ' + str(len(ones)), str(round((100 * len(ones.intersection(dict_check_flags))) / len(ones), 2)) + '%'], 
                                               [str(len(dict_check_flags)) + ' / ' + str(len(ones)), str(round(100 * (len(dict_check_flags) / len(ones)), 2)) + '%'], 
                                               [str(len(dict_check_flags.difference(ones))), str(round((100 * len(dict_check_flags.difference(ones))) / len(dict_check_flags), 2)) + '%']], 
                                               columns = ['Absolute Values', 'In percentage'], 
                                               index = ['Accuracy respect the confirmed cases', '#application cases vs. #standard cases', 'Number of not flagged cases'])
                st.table(summ_table)

            st.table(table_fin_res)
            if cat_sel_col == '-':
                st.table(pd.DataFrame(list_prob_cases, columns = ['Variable', 'Country', '% Value', 'Absolute values']))
            else:
                st.table(pd.DataFrame(list_prob_cases, columns = ['Variable', 'Country', 'Category', '% Value', 'Absolute values']))
            
            dict_trend = {'Strong decrease': [], 'Weak decrease': [], 'Undetermined trend': [], 'Weak increase': [], 'Strong increase': []}; set_trend = set()
            for inst in dict_check_flags:
                class_tr = int(table[table[con_checks_id_col] == inst]['Class trend'].unique()[0])
                if class_tr != 0:
                    dict_trend[list(dict_trend.keys())[class_tr-1]].append(inst)
                    if class_tr == 1 or class_tr == 3 or class_tr == 5:
                        set_trend.add(inst)
            trend_table = pd.DataFrame([len(v) for v in dict_trend.values()], index = dict_trend.keys(), columns = ['Number of institutions'])
            
            st.table(trend_table)
            if flag_radio == 'Yes':
                if flag_notes_on:
                    summ_table = pd.DataFrame([[str(len(twos.intersection(set_trend))) + ' over ' + str(len(twos)), str(round((100 * len(twos.intersection(set_trend))) / len(twos), 2)) + '%'], 
                                               [str(len(set_trend)) + ' / ' + str(len(ones.union(twos))), str(round(100 * (len(set_trend) / len(ones.union(twos))), 2)) + '%'], 
                                               [str(len(set_trend.difference(ones.union(twos)))), str(round((100 * len(set_trend.difference(ones.union(twos)))) / len(set_trend), 2)) + '%']], 
                                               columns = ['Absolute Values', 'In percentage'], 
                                               index = ['Accuracy respect the confirmed cases', '#application cases vs. #standard cases', 'Number of not flagged cases'])
                else:
                    summ_table = pd.DataFrame([[str(len(ones.intersection(set_trend))) + ' over ' + str(len(ones)), str(round((100 * len(ones.intersection(set_trend))) / len(ones), 2)) + '%'], 
                                               [str(len(set_trend)) + ' / ' + str(len(ones)), str(round(100 * (len(set_trend) / len(ones)), 2)) + '%'], 
                                               [str(len(set_trend.difference(ones))), str(round((100 * len(set_trend.difference(ones))) / len(set_trend), 2)) + '%']], 
                                               columns = ['Absolute Values', 'In percentage'], 
                                               index = ['Accuracy respect the confirmed cases', '#application cases vs. #standard cases', 'Number of not flagged cases'])
                st.table(summ_table)
            
            trend_type = st.selectbox('Choose the institution trend type you want to vizualize', list(dict_trend.keys()), 0)
            trend_inst = st.selectbox('Choose the institution you want to vizualize', dict_trend[trend_type])
            line_trend_ch_inst = px.line(table[table[con_checks_id_col] == trend_inst][[con_checks_features, 'Reference year']], x = 'Reference year', y = con_checks_features)
            line_trend_ch_inst.update_yaxes(range = [0, max(table[table[con_checks_id_col] == trend_inst][con_checks_features].values) + (.05 * max(table[table[con_checks_id_col] == trend_inst][con_checks_features].values))])
            st.plotly_chart(line_trend_ch_inst, use_container_width=True)
            
            cols_pr_inst = st.multiselect('Choose the variables', col_mul); dict_pr_inst = {}
            for col in cols_pr_inst:
                for id_inst in table[con_checks_id_col].unique():
                    # calculations of the geometric mean
                    inst = table[table[con_checks_id_col] == id_inst][col].values[::-1]
                    geo_mean_vec = np.delete(inst, np.where((inst == 0) | (np.isnan(inst))))
                    if geo_mean_vec.shape[0] != 0:
                        res_ind[id_inst] = math.pow(math.fabs(np.prod(geo_mean_vec)), 1/geo_mean_vec.shape[0])
                    else:
                        res_ind[id_inst] = np.nan
                        
                indices = pd.DataFrame(res_ind.values(), index = res_ind.keys(), columns = [col])
                indices.drop(index = set(indices[(pd.isna(indices[col])) | (indices[col] <= indices.quantile(0.02).values[0])].index), axis = 0, inplace = True)

                res = dict()
                # does the calculation with the delta+ and delta-minus for the multiannual checks and stores it into a dictionary 
                for id_inst in indices.index.values:
                    inst = table[(table[con_checks_id_col] == id_inst) & (-pd.isna(table[col]))][col].values
                    num_row = len(inst); delta_pos = list(); delta_neg = list()
                    for i in range(1, num_row):
                        if inst[num_row - i - 1] - inst[num_row - i] < 0:
                            delta_neg.append(round(inst[num_row - i - 1] - inst[num_row - i], 2))
                        else:
                            delta_pos.append(round(inst[num_row - i - 1] - inst[num_row - i], 2))
                    res[id_inst] = [delta_pos, delta_neg]

                DV = dict() # the dictionary in wich we'll store all the DV and further the DM values for the variability from years
                for key, value in res.items():
                    res_par = 0
                    if len(value[0]) != 0 and len(value[1]) != 0:
                        res_par = sum(value[0]) * sum(value[1])
                    DV[key] = round(math.fabs(res_par)/indices[col][key] ** 1.5, 3)
       
                DV_df = pd.DataFrame(DV.values(), index = DV.keys(), columns = [col])
                dict_check_flags = set(DV_df[DV_df[col] >= DV_df[col].quantile(0.95)].index)
            
                for inst in dict_check_flags:
                    if inst not in dict_pr_inst.keys():
                        dict_pr_inst[inst] = [col]
                    else:
                        dict_pr_inst[inst].append(col)
                
            dict_pr_inst = dict(sorted(dict_pr_inst.items(), key = lambda item: len(item[1]), reverse = True))
            dict_pr_inst = {k: [len(v), ' '.join(v)] for k, v in dict_pr_inst.items()}
            st.table(pd.DataFrame(dict_pr_inst.values(), index = dict_pr_inst.keys(), columns = ['# of problems', 'Probematic variables']).head(25))
            
            # part of confronting trends
            conf_trend_radio = st.radio("Do you want to compare trends?", ('Yes', 'No'), key = 'conf_trend_ratio')
            if conf_trend_radio == 'Yes':
                conf_trend_var = st.selectbox("Variables chosen for the consistency checks:", col_mul, key = 'conf_trend_var'); set_not_det = set()
                set_inc_inc = set(); set_inc_ukn = set(); set_inc_dec = set()
                set_ukn_inc = set(); set_ukn_ukn = set(); set_ukn_dec = set()
                set_dec_inc = set(); set_dec_ukn = set(); set_dec_dec = set()
                
                for var in table[table['Prob inst ' + con_checks_features] == 1][con_checks_id_col].unique():
                    inst = table[table[con_checks_id_col] == var][conf_trend_var].values[::-1]
                    geo_mean_vec = np.delete(inst, np.where((inst == 0) | (np.isnan(inst))))

                    # trend classification
                    if geo_mean_vec.shape[0] > 3:
                        mann_kend_res = mk.original_test(geo_mean_vec)
                        trend, p, tau = mann_kend_res.trend, mann_kend_res.p, mann_kend_res.Tau
                        if p <= p_value_trend_per/100 and tau >= 0:
                            if table[table[con_checks_id_col] == var]['Class trend'].unique()[0] > 3:
                                set_inc_inc.add(var)
                            if table[table[con_checks_id_col] == var]['Class trend'].unique()[0] == 3:
                                set_inc_ukn.add(var)
                            if table[table[con_checks_id_col] == var]['Class trend'].unique()[0] < 3:
                                set_inc_dec.add(var)
                        if p <= p_value_trend_per/100 and tau < 0:
                            if table[table[con_checks_id_col] == var]['Class trend'].unique()[0] > 3:
                                set_dec_inc.add(var)
                            if table[table[con_checks_id_col] == var]['Class trend'].unique()[0] == 3:
                                set_dec_ukn.add(var)
                            if table[table[con_checks_id_col] == var]['Class trend'].unique()[0] < 3:
                                set_dec_dec.add(var)
                        if p > p_value_trend_per/100:
                            if table[table[con_checks_id_col] == var]['Class trend'].unique()[0] > 3:
                                set_ukn_inc.add(var)
                            if table[table[con_checks_id_col] == var]['Class trend'].unique()[0] == 3:
                                set_ukn_ukn.add(var)
                            if table[table[con_checks_id_col] == var]['Class trend'].unique()[0] < 3:
                                set_ukn_dec.add(var)
                    else:
                        set_not_det.add(var)
                    
                table_conf_trend = [[len(set_inc_inc), len(set_inc_ukn), len(set_inc_dec)], 
                                    [len(set_ukn_inc), len(set_ukn_ukn), len(set_ukn_dec)], 
                                    [len(set_dec_inc), len(set_dec_ukn), len(set_dec_dec)]]
                st.table(pd.DataFrame(table_conf_trend, 
                                      index = ['(' + conf_trend_var + ') ' + 'Increasing', '(' + conf_trend_var + ') ' + 'Unknown', '(' + conf_trend_var + ') ' + 'Decreasing'], 
                                      columns = ['(' + con_checks_features + ') ' + 'Increasing', '(' + con_checks_features + ') ' + 'Unknown', '(' + con_checks_features + ') ' + 'Decreasing']))
                st.write('The number of institution that couldn\'t be classified because of lacking data: ' + str(len(set_not_det)))
            
            st.write('If you want to download the result file with all the issued flags you have first to choose at least the time column and then to clik on the following button:')
            left1, right1 = st.columns(2)
            with left1:
                time_col = st.selectbox("Select the variable from wich you want to extract the time values:", table.columns)
            with right1:
                descr_col = st.multiselect("Select the desciptive columns you want to add to the result dataset:", table.columns)

            t_col = [str(el) for el in sorted(table[time_col].unique())]; list_fin = []
            if flag_radio == 'Yes':
                df_cols = [con_checks_id_col] + descr_col + t_col + ['Variable', 'Trend', 'Existing flag', 'Detected case']
            else:
                df_cols = [con_checks_id_col] + descr_col + t_col + ['Variable', 'Trend', 'Detected case']
            for inst in sorted(list(table[con_checks_id_col].unique())):
                df_inst = table[table[con_checks_id_col] == inst]
                list_el = [inst]
                for col in descr_col:
                    list_el.append(df_inst[col].unique()[0])
                for t in t_col:
                    if df_inst[df_inst[time_col] == int(t)].shape[0] != 0:
                        list_el.append(df_inst[df_inst[time_col] == int(t)][con_checks_features].values[0])
                    else:
                        list_el.append(np.nan)
                list_el.append(con_checks_features)
                if df_inst['Class trend'].unique()[0] == 0:
                    list_el.append('Impossible to calculate')
                else:
                    list_el.append(list(dict_trend.keys())[df_inst['Class trend'].unique()[0]-1])
                if flag_radio == 'Yes':
                    if notes_col != '-':
                        if (inst not in ones) and (inst not in twos):
                            list_el.append(0)
                        if inst in ones:
                            list_el.append(1)
                        if inst in twos:
                            list_el.append(2)
                    else:
                        if inst not in ones:
                            list_el.append(0)
                        else:
                            list_el.append(1)
                list_el.append(df_inst['Prob inst ' + con_checks_features].unique()[0])
                list_fin.append(list_el)
            for i in range(len(list_fin)):
                if len(df_cols) != len(list_fin[i]):
                    st.write(list_fin[i])
            table_download = pd.DataFrame(list_fin, columns = df_cols)
            st.download_button(label = "Download data with lables", data = table_download.to_csv(index = None, sep = ';').encode('utf-8'), file_name = 'result.csv', mime = 'text/csv')
            

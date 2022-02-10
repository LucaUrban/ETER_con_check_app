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

    # selection boxes columns
    col_an = [col for col in list(table) if len(table[col].unique()) < 10 or is_numeric_dtype(table[col])]
    col_mul = [col for col in list(table) if is_numeric_dtype(table[col])]
    lis_check = [{'label': col, 'value': col} for col in col_mul if col != col_mul[0]]

    widget = st.selectbox("what is the widget you want to display:", ["Multidiannual Methodology", "Ratio Methodology"], 0)

    if widget == 'Ratio Methodology':
        con_checks_id_col = st.sidebar.selectbox("Index col", table.columns, 0)
        country_sel_col = st.sidebar.selectbox("Country selection column", ['-'] + list(table.columns), 0)
        cat_sel_col = st.sidebar.selectbox("Category selection column", ['-'] + list(table.columns), 0)
        flag_issue_quantile = st.sidebar.number_input("Insert the quantile that will issue the flag (S2 and S3)", 0.0, 40.0, 5.0, 0.1)
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
                    summ_table = pd.DataFrame([[str(len(twos.intersection(dict_check_flags[con_checks_feature]))) + ' over ' + str(len(twos)), str(round((100 * len(twos.intersection(dict_check_flags[con_checks_feature]))) / len(twos), 2)) + '%'], 
                                               [str(len(dict_check_flags[con_checks_feature])) + ' / ' + str(len(ones.union(twos))), str(round(100 * (len(dict_check_flags[con_checks_feature]) / len(ones.union(twos))), 2)) + '%'], 
                                               [str(len(dict_check_flags[con_checks_feature].difference(ones.union(twos)))), str(round((100 * len(dict_check_flags[con_checks_feature].difference(ones.union(twos)))) / len(dict_check_flags[con_checks_feature]), 2)) + '%']], 
                                               columns = ['Absolute Values', 'In percentage'], 
                                               index = ['Accuracy respect the confirmed cases', '#application cases vs. #standard cases', 'Number of not flagged cases'])
                else:
                    summ_table = pd.DataFrame([[str(len(ones.intersection(dict_check_flags[con_checks_feature]))) + ' over ' + str(len(ones)), str(round((100 * len(ones.intersection(dict_check_flags[con_checks_feature]))) / len(ones), 2)) + '%'], 
                                               [str(len(dict_check_flags[con_checks_feature])) + ' / ' + str(len(ones)), str(round(100 * (len(dict_check_flags[con_checks_feature]) / len(ones)), 2)) + '%'], 
                                               [str(len(dict_check_flags[con_checks_feature].difference(ones))), str(round((100 * len(dict_check_flags[con_checks_feature].difference(ones))) / len(dict_check_flags[con_checks_feature]), 2)) + '%']], 
                                               columns = ['Absolute Values', 'In percentage'], 
                                               index = ['Accuracy respect the confirmed cases', '#application cases vs. #standard cases', 'Number of not flagged cases'])
                st.table(summ_table)

            table_fin_res = pd.DataFrame(list_fin_res, index = [con_checks_feature + ' (' + str(cat) + ')' for cat in categories] + ['Total'], columns = countries + ['Total'])
            st.table(table_fin_res)
            st.table(pd.DataFrame(list_prob_cases, columns = ['Variable', 'Country', 'Category', '% Value', 'Absolute values']))

            if len(list(table['Class trend'].unique())) > 1:
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
                        list_el.append(df_inst[df_inst[time_col] == int(t)][con_checks_feature].values[0])
                    else:
                        list_el.append(np.nan)
                list_el.append(con_checks_feature)
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
                list_el.append(df_inst['Prob inst ' + con_checks_feature].unique()[0])
                list_fin.append(list_el)
            for i in range(len(list_fin)):
                if len(df_cols) != len(list_fin[i]):
                    st.write(list_fin[i])
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
            

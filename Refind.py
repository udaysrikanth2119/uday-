#!/usr/bin/env python
# coding: utf-8

# In[302]:


import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.cluster import KMeans
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings("ignore")
# Read the dataset
data = pd.read_csv("API_19_DS2_en_csv_v2_5361599.csv",skiprows=4)

# Filter relevant columns and rows
relevant_columns = ["Country Name", "Indicator Name",'1960','1961','1962','1963','1964','1965','1966','1967','1968','1969','1970','1971','1972','1973','1974','1975','1976','1977','1978','1979','1980','1981','1982','1983','1984','1985','1986','1987','1988','1989','1990','1991','1992','1993','1994','1995','1996','1997','1998','1999','2000','2001','2002','2003','2004','2005','2006','2007','2008','2009','2010','2011','2012','2013','2014',"2015", "2016", "2017", "2018", "2019", "2020"]
relevant_rows = data[data["Indicator Name"] == "Population, total"]
filtered_data = relevant_rows[relevant_columns]
filtered_data


# In[303]:


data_f = data
data_f1 = data_f.drop(columns=["Country Code", "Indicator Code"], axis=1)
# data_f1 = data_f1.set_index('Country Name')
data_f1


# In[304]:


data_f1.isnull().sum()
data2 = data_f1.fillna(0)
data2


# In[305]:


data1 = data2
# data1 = data2.drop(["Country Name","Indicator Name"],axis =1)
data1.set_index("Country Name", inplace=True)
data1 = data1.loc["United States"]
data1 = data1.reset_index(level="Country Name")
data1.groupby(["Indicator Name"])
data1 = data1.drop(["Unnamed: 66", '1960', '1961', '1962', '1963', '1964', '1965', '1966', '1967', '1968', '1969',
                    '1970', '1971', '1972', '1973', '1974', '1975', '1976', '1977', '1978', '1979',
                    '1980', '1981', '1982', '1983', '1984', '1985', '1986', '1987', '1988', '1989',
                    '1990', '1991', '1992', '1993', '1994', '1995', '1996', '1997', '1998', '1999', '2000', '2001',
                    '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013',
                    '2014', 'Country Name'], axis=1)

data1
data1 = data1.set_index('Indicator Name')
data1 = data1.transpose()
data1


# In[306]:


""" Tools to support clustering: correlation heatmap, normaliser and scale
(cluster centres) back to original scale, check for mismatching entries """


def map_corr(df, size=6):
    """Function creates heatmap of correlation matrix for each pair of
    columns in the dataframe.

    Input:
        df: pandas DataFrame
        size: vertical and horizontal size of the plot (in inch)

    The function does not have a plt.show() at the end so that the user
    can savethe figure.
    """

    import matplotlib.pyplot as plt  # ensure pyplot imported

    corr = df.corr()
    plt.figure(figsize=(size, size))
    # fig, ax = plt.subplots()
    plt.matshow(corr, cmap='coolwarm')
    # setting ticks to column names
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)

    plt.colorbar()
    # no plt.show() at the end



# In[307]:


data1 = data1.drop(['Urban population (% of total population)','Urban population','Urban population growth (annual %)','Population growth (annual %)','Poverty headcount ratio at $2.15 a day (2017 PPP) (% of population)','Prevalence of underweight, weight for age (% of children under 5)','Community health workers (per 1,000 people)','Mortality rate, under-5 (per 1,000 live births)','Primary completion rate, total (% of relevant age group)','School enrollment, primary and secondary (gross), gender parity index (GPI)','Agriculture, forestry, and fishing, value added (% of GDP)','CPIA public sector management and institutions cluster average (1=low to 6=high)','Ease of doing business rank (1=most business-friendly regulations)','Terrestrial and marine protected areas (% of total territorial area)','Marine protected areas (% of territorial waters)','Terrestrial protected areas (% of total land area)','Annual freshwater withdrawals, total (% of internal resources)','Annual freshwater withdrawals, total (billion cubic meters)','Population in urban agglomerations of more than 1 million (% of total population)','Population living in areas where elevation is below 5 meters (% of total population)','Urban population living in areas where elevation is below 5 meters (% of total population)','Rural population living in areas where elevation is below 5 meters (% of total population)','Droughts, floods, extreme temperatures (% of population, average 1990-2009)','GHG net emissions/removals by LUCF (Mt of CO2 equivalent)','Disaster risk reduction progress score (1-5 scale; 5=best)','SF6 gas emissions (thousand metric tons of CO2 equivalent)','PFC gas emissions (thousand metric tons of CO2 equivalent)','Nitrous oxide emissions (% change from 1990)','Nitrous oxide emissions (thousand metric tons of CO2 equivalent)','Methane emissions (% change from 1990)','Methane emissions (kt of CO2 equivalent)','HFC gas emissions (thousand metric tons of CO2 equivalent)','Total greenhouse gas emissions (kt of CO2 equivalent)','Other greenhouse gas emissions (% change from 1990)','Other greenhouse gas emissions, HFC, PFC and SF6 (thousand metric tons of CO2 equivalent)','CO2 emissions from solid fuel consumption (% of total)','CO2 emissions from solid fuel consumption (kt)','CO2 emissions (kg per 2017 PPP $ of GDP)','CO2 emissions (kg per PPP $ of GDP)','CO2 emissions (metric tons per capita)','CO2 emissions from liquid fuel consumption (kt)','CO2 emissions (kt)','CO2 emissions (kg per 2015 US$ of GDP)','CO2 emissions from gaseous fuel consumption (% of total)','CO2 emissions from gaseous fuel consumption (kt)','CO2 intensity (kg per kg of oil equivalent energy use)','Energy use (kg of oil equivalent per capita)','Electric power consumption (kWh per capita)','Energy use (kg of oil equivalent) per $1,000 GDP (constant 2017 PPP)','Renewable energy consumption (% of total final energy consumption)','Electricity production from renewable sources, excluding hydroelectric (% of total)','Electricity production from renewable sources, excluding hydroelectric (kWh)','Renewable electricity output (% of total electricity output)','Electricity production from oil sources (% of total)','Electricity production from nuclear sources (% of total)','Electricity production from hydroelectric sources (% of total)','Electricity production from coal sources (% of total)','Access to electricity (% of population)','Foreign direct investment, net inflows (% of GDP)','Cereal yield (kg per hectare)','Average precipitation in depth (mm per year)','Agricultural irrigated land (% of total agricultural land)','Land area where elevation is below 5 meters (% of total land area)','Urban land area where elevation is below 5 meters (% of total land area)','Urban land area where elevation is below 5 meters (sq. km)','Rural land area where elevation is below 5 meters (% of total land area)','Rural land area where elevation is below 5 meters (sq. km)','Arable land (% of land area)','Agricultural land (% of land area)','Agricultural land (sq. km)','Total greenhouse gas emissions (% change from 1990)'],axis=1)

map_corr(data1)
plt.show()


# In[308]:


relevant_columns1 = ['1960','1961','1962','1963','1964','1965','1966','1967','1968','1969','1970','1971','1972','1973','1974','1975','1976','1977','1978','1979','1980','1981','1982','1983','1984','1985','1986','1987','1988','1989','1990','1991','1992','1993','1994','1995','1996','1997','1998','1999','2000','2001','2002','2003','2004','2005','2006','2007','2008','2009','2010','2011','2012','2013','2014',"2015", "2016", "2017", "2018", "2019", "2020"]
new_df = filtered_data[relevant_columns1]
new_df


# In[308]:





# In[309]:


new_df = new_df.fillna(0)

#K means model with 5 clusters
c_model = KMeans(n_clusters=5)
#training the model
clusters = c_model.fit_predict(new_df)
#creating a blank dataframe
new_data = pd.DataFrame()
#storing the value of clusters generated from the k-means model into the column "Clusters" of this dataframe
new_data["Clusters"] = clusters
#checking unique cluster values
unq_value = new_data["Clusters"].unique()
unq_value


# In[310]:


index = list(filtered_data.index.values)
#setting the index of new_data dataframe as that of the original_version1 so that they can be merged later on
new_data["index"]=index
new_data.set_index(["index"], drop=True)


# In[311]:


filtered_data["index"]=index
filtered_data.set_index(["index"], drop=True)


# In[312]:


#merging both the datasets
filtered_data = pd.merge(filtered_data, new_data)
#dropping the unnecessary column index
filtered_data.drop(columns=["index"], inplace=True)
#Number of values in each cluster
filtered_data.Clusters.value_counts()


# In[313]:


#computing the centroids for each cluster
centroid_vals = c_model.cluster_centers_
data_blance = []
data2 = []
#getting the centroid value of only 2017 and 2021
for i in centroid_vals:
    for j in range(len(i)):
        x = i[15]
        y = i[60]
    data_blance.append(i[15])
    data2.append(i[60])


# In[314]:


#defining the colors in a list
colors = ['#fc0305', '#fa05f6', '#9b04ff', '#0543ed', "#020925"]
#mapping the colors according to unique values in the column Clusters
filtered_data['c'] = filtered_data.Clusters.map(
    {0: colors[0], 1: colors[1], 2: colors[2], 3: colors[3], 4: colors[4]})
#initiating a figure
fig, ax = plt.subplots(1, figsize=(15, 8))
#plotting a scatter plot of data
plt.scatter(filtered_data["1980"], filtered_data["2020"], c=filtered_data.c, alpha=0.7, s=40)
#plotting a scatter plot of centroids
plt.scatter(data_blance, data2, marker='^', facecolor=colors, edgecolor="black", s=100)
#getting the legend for data
legend_elements = [Line2D([0], [0], marker='o', color='w', label='Cluster or C{}'.format(i + 1),
                          markerfacecolor=mcolor, markersize=5) for i, mcolor in enumerate(colors)]
#getting the legend for centroids
centroid_legend = [Line2D([0], [0], marker='^', color='w', label='Centroid of C{}'.format(i + 1),
                          markerfacecolor=mcolor, markeredgecolor="black", markersize=10) for i, mcolor in
                   enumerate(colors)]
#final legend elements
legend_elements.extend(centroid_legend)
#setting the legend
plt.legend(handles=legend_elements, loc='upper right', title="Clusters", fontsize=10, bbox_to_anchor=(1.15, 1))
#setting xlabel, ylabel and title
plt.xlabel("1980 data in %", fontsize='18')
plt.ylabel("2020 data in %", fontsize='18')
plt.title("Basic K-means clustering on the basis of population growth (annual %)", fontsize='20')
plt.show()

# In[315]:


#Building a function to fit the model using the curve_fit of scipy
def new_datafit(frame, col_1, col_2):
    """This function takes in a dataframe and columns to fit the model using the curve_fit methodology of the scipy library.

    Parameters
    ----------
    csv_fileframe : pandas dataframe
        The name of the pandas dataframe.
    col_1 : str
        The name of the column of dataframe to be taken as x.
    col_2 : str
        The name of the column of dataframe to be taken as y.

    Returns
    -------
    errors : numpy array
        An array containing the errors or ppot.
    covariance : numpy array
        An array containing the covariances.
    """
    col1_data = frame[col_1]
    col2_data = frame[col_2]

    def mod_func(x, m, b):
        return m*x+b

    #calling curve_fit
    errors,covariance = curve_fit(mod_func, col1_data, col2_data)
    plt.figure(figsize=(15,8))
    #plotting the data
    plt.plot(col1_data, col2_data, "bo", label="data", color ="g")
    #plotting the best fitted line
    plt.plot(col1_data, mod_func(col1_data, *errors), "b-", label="Best Fit")
    plt.xlabel(col_1)
    plt.ylabel(col_2)
    plt.legend(bbox_to_anchor=(1,1))
    plt.title("Plotting best fit for population growth (annual %)")
    plt.show()
    return errors, covariance


# In[316]:


""" Module errors. It provides the function err_ranges which calculates upper
and lower limits of the confidence interval. """

import numpy as np


def err_ranges(x, func, param, sigma):
    """
    Calculates the upper and lower limits for the function, parameters and
    sigmas for single value or array x. Functions values are calculated for
    all combinations of +/- sigma and the minimum and maximum is determined.
    Can be used for all number of parameters and sigmas >=1.

    This routine can be used in assignment programs.
    """

    import itertools as iter

    # initiate arrays for lower and upper limits
    lower = func(x, *param)
    upper = lower

    uplow = []   # list to hold upper and lower limits for parameters
    for p, s in zip(param, sigma):
        pmin = p - s
        pmax = p + s
        uplow.append((pmin, pmax))

    pmix = list(iter.product(*uplow))

    for p in pmix:
        y = func(x, *p)
        lower = np.minimum(lower, y)
        upper = np.maximum(upper, y)

    return lower, upper



# In[317]:


#calling the function
new_datafit(new_df, "1980", "2020")


# In[317]:





# In[317]:





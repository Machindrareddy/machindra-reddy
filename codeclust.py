# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 13:30:29 2024

@author: Machindra
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import scipy.optimize as opt

def load_and_plot_clusters(file_path):
    # Load the dataset
    data = pd.read_excel(file_path)
    
    # Selected countries and indicators
    selected_countries = [
        "Argentina", "Austria", "China", "Germany", "France",
        "United Kingdom", "Indonesia", "India", "Japan",
        "New Zealand", "Singapore", "United States"
    ]

    selected_indicators = [
        "Population growth (annual %)",
        "Agricultural land (% of land area)"
    ]

    # Filtering the data
    years_range = list(range(2010, 2021))
    filtered_data = data[
        (data["Country Name"].isin(selected_countries)) &
        (data["Indicator Name"].isin(selected_indicators))
    ][["Country Name", "Indicator Name"] + years_range]

    # Melting the data for clustering
    data_melted = pd.melt(filtered_data, id_vars=["Country Name", "Indicator Name"],
                          var_name="Year", value_name="Value")

    # Preparing population and agricultural data
    population_data = data_melted[data_melted["Indicator Name"] == "Population growth (annual %)"].drop('Indicator Name', axis=1)
    agricultural_data = data_melted[data_melted["Indicator Name"] == "Agricultural land (% of land area)"].drop('Indicator Name', axis=1)

    population_data.rename(columns={"Value": "Population growth (annual %)"}, inplace=True)
    agricultural_data.rename(columns={"Value": "Agricultural land (% of land area)"}, inplace=True)

    # Merging and normalizing data
    merged_data = pd.merge(population_data, agricultural_data, on=["Country Name", "Year"])
    min_max_scaler = MinMaxScaler()
    normalized_data = merged_data.copy()
    normalized_data[["Population growth (annual %)", "Agricultural land (% of land area)"]] = min_max_scaler.fit_transform(merged_data[["Population growth (annual %)", "Agricultural land (% of land area)"]])

    # KMeans clustering
    kmeans = KMeans(n_clusters=5, random_state=0)
    kmeans.fit(normalized_data[['Population growth (annual %)', 'Agricultural land (% of land area)']])

    # Assigning clusters
    normalized_data['Cluster'] = kmeans.labels_

    # Define different colors for clusters
    cluster_colors = ['red', 'blue', 'green', 'orange', 'purple']

    # Rename clusters for legend
    cluster_name_mapping = {0: 'Group A', 1: 'Group B', 2: 'Group C', 3: 'Group D', 4: 'Group E'}
    normalized_data['Cluster'] = normalized_data['Cluster'].map(cluster_name_mapping)

    # Plotting clusters
    plt.figure(figsize=(12, 6))
    sns.scatterplot(data=normalized_data, x='Population growth (annual %)', y='Agricultural land (% of land area)', hue='Cluster', palette=cluster_colors, style='Cluster', legend='full')
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='+', s=100, color='black', label='Cluster Centers')

    for cluster_num, cluster_name in cluster_name_mapping.items():
        cluster_points = normalized_data[normalized_data['Cluster'] == cluster_name]
        cluster_center = kmeans.cluster_centers_[cluster_num]
        max_distance = np.max(np.linalg.norm(cluster_points[['Population growth (annual %)', 'Agricultural land (% of land area)']].values - cluster_center, axis=1))
        circle = plt.Circle(cluster_center, max_distance, color=cluster_colors[cluster_num], fill=False, linestyle='dotted', alpha=0.5)
        plt.gca().add_artist(circle)

    plt.title('Population growth vs Agricultural land')
    plt.xlabel('Population growth (annual %)')
    plt.ylabel('Agricultural land (% of land area)')
    plt.legend(title='Cluster', loc='upper right', bbox_to_anchor=(1.3, 1))
    plt.show()

    # Polynomial Fitting Section
    # Filter data for the two indicators
    population_growth_data = data[data['Indicator Name'] == 'Population growth (annual %)']
    agricultural_land_data = data[data['Indicator Name'] == 'Agricultural land (% of land area)']

    # Reshape the data for each indicator
    def reshape_data(data):
        data_reshaped = data.drop(['Indicator Name'], axis=1).set_index('Country Name').T
        data_reshaped.index = data_reshaped.index.astype(int)
        return data_reshaped

    population_growth_reshaped = reshape_data(population_growth_data)
    agricultural_land_reshaped = reshape_data(agricultural_land_data)

    # Define the polynomial model fitting function
    def polynomial_with_error(x, a, b, c, d, e):
        x = x - 2000
        return a + b * x + c * x**2 + d * x**3 + e * x**4

    # Partial Derivatives for Polynomial Model
    dfunc = [
        lambda x, a, b, c, d, e: 1,
        lambda x, a, b, c, d, e: (x - 2000),
        lambda x, a, b, c, d, e: (x - 2000)**2,
        lambda x, a, b, c, d, e: (x - 2000)**3,
        lambda x, a, b, c, d, e: (x - 2000)**4
    ]

    # Confidence Interval Calculation Function
    def confidence_interval(x, params, covar, func):
        pred = func(x, *params)
        J = np.array([[df(x, *params) for df in dfunc] for x in x])
        pred_se = np.sqrt(np.diag(J @ covar @ J.T))
        ci = 1.96 * pred_se  # 95% confidence interval
        return pred - ci, pred + ci

    # Sample countries to plot
    sample_countries = ['Germany', 'France', 'New Zealand', 'Japan']

    # Define custom colors for lines, points, and confidence intervals
    colors = ['navy', 'maroon', 'teal', 'gold']

    # Loop over each country for Population Growth
    for i, country in enumerate(sample_countries):
        # Check if the country data is available
        if country in population_growth_reshaped.columns:
            # Fit the Polynomial Model
            country_data = population_growth_reshaped[country].dropna()
            param_poly, covar_poly = opt.curve_fit(
                polynomial_with_error, 
                country_data.index, 
                country_data.values,
                maxfev=10000  # Increase the number of function evaluations
            )

            # Generate Predictions and Confidence Intervals
            year_range = np.arange(2000, 2026)  # Extended to include 2026
            low_poly, up_poly = confidence_interval(year_range, param_poly, covar_poly, polynomial_with_error)

            # Prediction for 2026
            prediction_2026 = polynomial_with_error(2026, *param_poly)

            # Create a new figure for each country
            plt.figure(figsize=(10, 5))

            # Plotting for Population Growth
            plt.plot(country_data.index, country_data, label=f"Actual Data - {country}", marker="o", color=colors[i % len(colors)])
            plt.plot(year_range, polynomial_with_error(year_range, *param_poly), label=f"Polynomial Fit - {country}", color=colors[(i + 1) % len(colors)])
            plt.fill_between(year_range, low_poly, up_poly, color=colors[(i + 2) % len(colors)], alpha=0.5, label=f"95% Confidence Interval - {country}")
            plt.plot(2026, prediction_2026, marker='o', markersize=8, label=f'Prediction for 2026: {prediction_2026:.2f}', color=colors[(i + 3) % len(colors)])
            plt.title(f"Population growth (annual %) - {country}")
            plt.xlabel("Year")
            plt.ylabel("Population growth (annual %)")
            plt.legend()
            plt.tight_layout()
            plt.show()
load_and_plot_clusters('world_bank_data_New.xlsx')

# name: Hyperauthorship_threshold_method.py
# author: Ly Dinh
# date created: 04/19/2023
# note: Input file should include the counts of the number of co-authors for each paper in your dataset

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import pandas as pd

# Define Chebyshev's inequality function
def chebyshev_inequality(count):
    return 1 / count**2

# Load data
file = "your path here"
author_counts = pd.read_csv(file, keep_default_na=False)
counts = author_counts['Total_Authors']
counts = counts.sort_values()
counts_array = np.asarray(counts)

# Calculate summary statistics
mean = counts.mean()
median = counts.median()
mode = counts.mode()
min_val = counts.min()
max_val = counts.max()
sd = counts.std()

# Calculate upper and lower bounds for different k values
def calculate_bounds(k):
    upper_bound = mean + k * sd
    lower_bound = mean - k * sd
    return upper_bound, lower_bound

# Calculate probabilities using Chebyshev's inequality
def calculate_cheb_probs(counts):
    p_cheb = np.zeros(len(counts))
    for i in range(len(counts)):
        p_cheb[i] = chebyshev_inequality(counts[i])
    return p_cheb

# Calculate probabilities using the cumulative normal distribution
def calculate_norm_probs(counts_array):
    p_norm = np.zeros(len(counts_array))
    for i in range(len(counts_array)):
        p_norm[i] = (1 - norm.cdf(counts_array[i])) * 2
    return p_norm

# Plot probabilities
def plot_probs(counts_array, p_cheb, p_norm):
    plt.figure(figsize=(20,10))
    plt.plot(counts_array, p_cheb, '-o')
    plt.plot(counts_array, p_norm, '-o')
    plt.xlabel('Number of authors', fontsize=20)
    plt.ylabel('Probability', fontsize=20)
    plt.legend(["Estimated by Chebyshev's Inequality","Estimated by Cumulative Normal Distribution"], fontsize=20)
    plt.savefig('Chebyshev.pdf')
    plt.show()

# Calculate frequency of author counts and plot
def plot_author_frequency(counts):
    plt.figure(figsize=(40,20))
    df = pd.DataFrame({'num_authors': counts})
    df.groupby('num_authors', as_index=True).size().plot(kind='bar', color='black')
    plt.xlabel('Number of co-authors (per paper)', fontsize=50)
    plt.ylabel('Frequency', fontsize=50)
    plt.savefig("author_dist_cutoff.pdf", format="pdf", bbox_inches="tight")
    plt.show()

# Calculate cumulative frequency and percent
def calculate_cumulative(df):
    df['Cumulative Frequency'] = df['num_authors'].cumsum()
    df['Cumulative Percent'] = round((df['num_authors'].cumsum() / df['num_authors'].sum()) * 100)
    return df

# Identify the cutoff point for 90% cumulative percent
def identify_cutoff(df):
    cutoff_point = df.loc[df['Cumulative Percent'] == 90.0]
    return cutoff_point

# Main function
def main():
    # Calculate summary statistics
    print(f"Mean: {mean}")
    print(f"Median: {median}")
    print(f"Mode: {mode}")
    print(f"Min: {min_val}")
    print(f"Max: {max_val}")
    print(f"Standard Deviation: {sd}")

    # Example usage
    k_values = [2, 3, 4]
    for k in k_values:
        upper_bound, lower_bound = calculate_bounds(k)
        print(f"Upper bound for k={k}: {upper_bound}")
        print(f"Lower bound for k={k}: {lower_bound}")

    # Calculate probabilities
    p_cheb = calculate_cheb_probs(counts)
    p_norm = calculate_norm_probs(counts_array)

    # Plot probabilities
    plot_probs(counts_array, p_cheb, p_norm)

    # Plot author frequency
    plot_author_frequency(counts)

    # Calculate cumulative frequency and percent
    df = calculate_cumulative(pd.DataFrame({'num_authors': counts}))

    # Identify cutoff point for 90% cumulative percent
    cutoff_point = identify_cutoff(df)
    print("Cutoff point for 90% cumulative percent:")
    print(cutoff_point)

if __name__ == "__main__":
    main()

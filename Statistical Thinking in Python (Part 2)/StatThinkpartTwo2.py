
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

#%%
plt.style.use('classic')
plt.style.use('ggplot')

#%%

rainfall=np.array([  875.5,   648.2,   788.1,   940.3,   491.1,   743.5,   730.1,
                   686.5,   878.8,   865.6,   654.9,   831.5,   798.1,   681.8,
                   743.8,   689.1,   752.1,   837.2,   710.6,   749.2,   967.1,
                   701.2,   619. ,   747.6,   803.4,   645.6,   804.1,   787.4,
                   646.8,   997.1,   774. ,   734.5,   835. ,   840.7,   659.6,
                   828.3,   909.7,   856.9,   578.3,   904.2,   883.9,   740.1,
                   773.9,   741.4,   866.8,   871.1,   712.5,   919.2,   927.9,
                   809.4,   633.8,   626.8,   871.3,   774.3,   898.8,   789.6,
                   936.3,   765.4,   882.1,   681.1,   661.3,   847.9,   683.9,
                   985.7,   771.1,   736.6,   713.2,   774.5,   937.7,   694.5,
                   598.2,   983.8,   700.2,   901.3,   733.5,   964.4,   609.3,
                   1035.2,   718. ,   688.6,   736.8,   643.3,  1038.5,   969. ,
                   802.7,   876.6,   944.7,   786.6,   770.4,   808.6,   761.3,
                   774.2,   559.3,   674.2,   883.6,   823.9,   960.4,   877.8,
                   940.6,   831.8,   906.2,   866.5,   674.1,   998.1,   789.3,
                   915. ,   737.1,   763. ,   666.7,   824.5,   913.8,   905.1,
                   667.8,   747.4,   784.7,   925.4,   880.2,  1086.9,   764.4,
                   1050.1,   595.2,   855.2,   726.9,   785.2,   948.8,   970.6,
                   896. ,   618.4,   572.4,  1146.4,   728.2,   864.2,   793. ])

#%%


def ecdf(data):
    """Compute ECDF for a one-dimensional array of measurements."""
    # Number of data points: n
    n=len(data)

    # x-data for the ECDF: x
    x=np.sort(data)

    # y-data for the ECDF: y
    y = np.arange(1, n + 1) / n

    return x, y


#%%

for _ in range(50):
    # Generate bootstrap sample: bs_sample
    bs_sample = np.random.choice(rainfall, size=len(rainfall))

    # Compute and plot ECDF from bootstrap sample
    x, y = ecdf(bs_sample)
    _ = plt.plot(x, y, marker='.', linestyle='none',
                 color='gray', alpha=0.1)

# Compute and plot ECDF from original data
x, y = ecdf(rainfall)
_ = plt.plot(x, y, marker='.')

# Make margins and label axes
plt.margins(0.02)
_ = plt.xlabel('yearly rainfall (mm)')
_ = plt.ylabel('ECDF')

# Show the plot
plt.show()

#%%

def bootstrap_replicate_1d(data, func):
    return func(np.random.choice(data, size=len(data)))


#%%


def draw_bs_reps(data, func, size=1):
    """Draw bootstrap replicates."""

    # Initialize array of replicates: bs_replicates
    bs_replicates = np.empty(size)

    # Generate replicates
    for i in range(size):
        bs_replicates[i] = bootstrap_replicate_1d(data,func)

    return bs_replicates

#%%
    
# Take 10,000 bootstrap replicates of the mean: bs_replicates
bs_replicates = draw_bs_reps(rainfall,np.mean,10000)

# Compute and print SEM
sem = np.std(rainfall) / np.sqrt(len(rainfall))
print(sem)

# Compute and print standard deviation of bootstrap replicates
bs_std = np.std(bs_replicates)
print(bs_std)

# Make a histogram of the results
_ = plt.hist(bs_replicates, bins=50,density=True)
_ = plt.xlabel('mean annual rainfall (mm)')
_ = plt.ylabel('PDF')

# Show the plot
plt.show()


#%%

np.round(np.percentile(bs_replicates,[2.5,97.5]))

#%%


# Generate 10,000 bootstrap replicates of the variance: bs_replicates
bs_replicates = draw_bs_reps(rainfall,np.var,10000)

# Put the variance in units of square centimeters
bs_replicates=bs_replicates/100

# Make a histogram of the results
_ = plt.hist(bs_replicates, density=True, bins=50)
_ = plt.xlabel('variance of annual rainfall (sq. cm)')
_ = plt.ylabel('PDF')

# Show the plot
plt.show()

#%%


nohitter_times=np.array([ 843, 1613, 1101,  215,  684,  814,  278,  324,  161,  219,  545,
                         715,  966,  624,   29,  450,  107,   20,   91, 1325,  124, 1468,
                         104, 1309,  429,   62, 1878, 1104,  123,  251,   93,  188,  983,
                         166,   96,  702,   23,  524,   26,  299,   59,   39,   12,    2,
                         308, 1114,  813,  887,  645, 2088,   42, 2090,   11,  886, 1665,
                         1084, 2900, 2432,  750, 4021, 1070, 1765, 1322,   26,  548, 1525,
                         77, 2181, 2752,  127, 2147,  211,   41, 1575,  151,  479,  697,
                         557, 2267,  542,  392,   73,  603,  233,  255,  528,  397, 1529,
                         1023, 1194,  462,  583,   37,  943,  996,  480, 1497,  717,  224,
                         219, 1531,  498,   44,  288,  267,  600,   52,  269, 1086,  386,
                         176, 2199,  216,   54,  675, 1243,  463,  650,  171,  327,  110,
                         774,  509,    8,  197,  136,   12, 1124,   64,  380,  811,  232,
                         192,  731,  715,  226,  605,  539, 1491,  323,  240,  179,  702,
                         156,   82, 1397,  354,  778,  603, 1001,  385,  986,  203,  149,
                         576,  445,  180, 1403,  252,  675, 1351, 2983, 1568,   45,  899,
                         3260, 1025,   31,  100, 2055, 4043,   79,  238, 3931, 2351,  595,
                         110,  215,    0,  563,  206,  660,  242,  577,  179,  157,  192,
                         192, 1848,  792, 1693,   55,  388,  225, 1134, 1172, 1555,   31,
                         1582, 1044,  378, 1687, 2915,  280,  765, 2819,  511, 1521,  745,
                         2491,  580, 2072, 6450,  578,  745, 1075, 1103, 1549, 1520,  138,
                         1202,  296,  277,  351,  391,  950,  459,   62, 1056, 1128,  139,
                         420,   87,   71,  814,  603, 1349,  162, 1027,  783,  326,  101,
                         876,  381,  905,  156,  419,  239,  119,  129,  467])


#%%
    
    
# Draw bootstrap replicates of the mean no-hitter time (equal to tau): bs_replicates
bs_replicates = draw_bs_reps(nohitter_times,np.mean,10000)

# Compute the 95% confidence interval: conf_int
conf_int = np.percentile(bs_replicates,[2.5,97.5])

# Print the confidence interval
print('95% confidence interval =', conf_int, 'games')

# Plot the histogram of the replicates
_ = plt.hist(bs_replicates, bins=50, density=True)
_ = plt.xlabel(r'$\tau$ (games)')
_ = plt.ylabel('PDF')

# Show the plot
plt.show()



#%%

def draw_bs_pairs_linreg(x, y, size=1):
    """Perform pairs bootstrap for linear regression."""

    # Set up array of indices to sample from: inds
    inds = np.arange(len(x))

    # Initialize replicates: bs_slope_reps, bs_intercept_reps
    bs_slope_reps = np.empty(size)
    bs_intercept_reps = np.empty(size)

    # Generate replicates
    for i in range(size):
        bs_inds = np.random.choice(inds, size=len(inds))
        bs_x, bs_y = x[bs_inds], y[bs_inds]
        bs_slope_reps[i], bs_intercept_reps[i] = np.polyfit(bs_x,bs_y,1)

    return bs_slope_reps, bs_intercept_reps


#%%

fertility=np.array([ 1.769,  2.682,  2.077,  2.132,  1.827,  3.872,  2.288,  5.173,
                    1.393,  1.262,  2.156,  3.026,  2.033,  1.324,  2.816,  5.211,
                    2.1  ,  1.781,  1.822,  5.908,  1.881,  1.852,  1.39 ,  2.281,
                    2.505,  1.224,  1.361,  1.468,  2.404,  5.52 ,  4.058,  2.223,
                    4.859,  1.267,  2.342,  1.579,  6.254,  2.334,  3.961,  6.505,
                    2.53 ,  2.823,  2.498,  2.248,  2.508,  3.04 ,  1.854,  4.22 ,
                    5.1  ,  4.967,  1.325,  4.514,  3.173,  2.308,  4.62 ,  4.541,
                    5.637,  1.926,  1.747,  2.294,  5.841,  5.455,  7.069,  2.859,
                    4.018,  2.513,  5.405,  5.737,  3.363,  4.89 ,  1.385,  1.505,
                    6.081,  1.784,  1.378,  1.45 ,  1.841,  1.37 ,  2.612,  5.329,
                    5.33 ,  3.371,  1.281,  1.871,  2.153,  5.378,  4.45 ,  1.46 ,
                    1.436,  1.612,  3.19 ,  2.752,  3.35 ,  4.01 ,  4.166,  2.642,
                    2.977,  3.415,  2.295,  3.019,  2.683,  5.165,  1.849,  1.836,
                    2.518,  2.43 ,  4.528,  1.263,  1.885,  1.943,  1.899,  1.442,
                    1.953,  4.697,  1.582,  2.025,  1.841,  5.011,  1.212,  1.502,
                    2.516,  1.367,  2.089,  4.388,  1.854,  1.748,  2.978,  2.152,
                    2.362,  1.988,  1.426,  3.29 ,  3.264,  1.436,  1.393,  2.822,
                    4.969,  5.659,  3.24 ,  1.693,  1.647,  2.36 ,  1.792,  3.45 ,
                    1.516,  2.233,  2.563,  5.283,  3.885,  0.966,  2.373,  2.663,
                    1.251,  2.052,  3.371,  2.093,  2.   ,  3.883,  3.852,  3.718,
                    1.732,  3.928])



illiteracy=np.array([  9.5,  49.2,   1. ,  11.2,   9.8,  60. ,  50.2,  51.2,   0.6,
                     1. ,   8.5,   6.1,   9.8,   1. ,  42.2,  77.2,  18.7,  22.8,
                     8.5,  43.9,   1. ,   1. ,   1.5,  10.8,  11.9,   3.4,   0.4,
                     3.1,   6.6,  33.7,  40.4,   2.3,  17.2,   0.7,  36.1,   1. ,
                     33.2,  55.9,  30.8,  87.4,  15.4,  54.6,   5.1,   1.1,  10.2,
                     19.8,   0. ,  40.7,  57.2,  59.9,   3.1,  55.7,  22.8,  10.9,
                     34.7,  32.2,  43. ,   1.3,   1. ,   0.5,  78.4,  34.2,  84.9,
                     29.1,  31.3,  18.3,  81.8,  39. ,  11.2,  67. ,   4.1,   0.2,
                     78.1,   1. ,   7.1,   1. ,  29. ,   1.1,  11.7,  73.6,  33.9,
                     14. ,   0.3,   1. ,   0.8,  71.9,  40.1,   1. ,   2.1,   3.8,
                     16.5,   4.1,   0.5,  44.4,  46.3,  18.7,   6.5,  36.8,  18.6,
                     11.1,  22.1,  71.1,   1. ,   0. ,   0.9,   0.7,  45.5,   8.4,
                     0. ,   3.8,   8.5,   2. ,   1. ,  58.9,   0.3,   1. ,  14. ,
                     47. ,   4.1,   2.2,   7.2,   0.3,   1.5,  50.5,   1.3,   0.6,
                     19.1,   6.9,   9.2,   2.2,   0.2,  12.3,   4.9,   4.6,   0.3,
                     16.5,  65.7,  63.5,  16.8,   0.2,   1.8,   9.6,  15.2,  14.4,
                     3.3,  10.6,  61.3,  10.9,  32.2,   9.3,  11.6,  20.7,   6.5,
                     6.7,   3.5,   1. ,   1.6,  20.5,   1.5,  16.7,   2. ,   0.9])


#%%
    
# Generate replicates of slope and intercept using pairs bootstrap
bs_slope_reps, bs_intercept_reps = draw_bs_pairs_linreg(illiteracy,fertility,1000)

# Compute and print 95% CI for slope
print(np.percentile(bs_slope_reps, [2.5,97.5]))

# Plot the histogram
_ = plt.hist(bs_slope_reps, bins=50, density=True)
_ = plt.xlabel('slope')
_ = plt.ylabel('PDF')
plt.show()


#%%

# Generate array of x-values for bootstrap lines: x
x = np.array([0,100])

# Plot the bootstrap lines
for i in range(100):
    _ = plt.plot(x, bs_slope_reps[i]*x + bs_intercept_reps[i],
                 linewidth=0.5, alpha=0.2, color='red')

# Plot the data
_ = plt.plot(illiteracy,fertility,marker='.',linestyle='none')

# Label axes, set the margins, and show the plot
_ = plt.xlabel('illiteracy')
_ = plt.ylabel('fertility')
plt.margins(0.02)
plt.show()


#%%




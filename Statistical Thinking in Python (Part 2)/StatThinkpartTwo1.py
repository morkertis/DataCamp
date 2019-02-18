
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

#%%
plt.style.use('classic')
plt.style.use('ggplot')


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

# Seed random number generator
np.random.seed(42)

# Compute mean no-hitter time: tau
tau = np.mean(nohitter_times)

# Draw out of an exponential distribution with parameter tau: inter_nohitter_time
inter_nohitter_time = np.random.exponential(tau, 100000)

# Plot the PDF and label axes
_ = plt.hist(inter_nohitter_time,
             bins=50, density=True, histtype='step')
_ = plt.xlabel('Games between no-hitters')
_ = plt.ylabel('PDF')

# Show the plot
plt.show()



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
    
# Create an ECDF from real data: x, y
x, y = ecdf(nohitter_times)

# Create a CDF from theoretical samples: x_theor, y_theor
x_theor, y_theor = ecdf(inter_nohitter_time)

# Overlay the plots
plt.plot(x_theor, y_theor)
plt.plot(x, y, marker='.', linestyle='none')

# Margins and axis labels
plt.margins(0.02)
plt.xlabel('Games between no-hitters')
plt.ylabel('CDF')

# Show the plot
plt.show()


#%%

# Plot the theoretical CDFs
plt.plot(x_theor, y_theor)
plt.plot(x, y, marker='.', linestyle='none')
plt.margins(0.02)
plt.xlabel('Games between no-hitters')
plt.ylabel('CDF')

# Take samples with half tau: samples_half
samples_half = np.random.exponential(tau/2,10000)

# Take samples with double tau: samples_double
samples_double = np.random.exponential(tau*2,10000)

# Generate CDFs from these samples
x_half, y_half = ecdf(samples_half)
x_double, y_double = ecdf(samples_double)

# Plot these CDFs as lines
_ = plt.plot(x_half, y_half)
_ = plt.plot(x_double, y_double)

# Show the plot
plt.show()

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

def pearson_r(x, y):
    """Compute Pearson correlation coefficient between two arrays."""
    # Compute correlation matrix: corr_mat
    corr_mat=np.corrcoef(x,y)

    # Return entry [0,1]
    return corr_mat[0,1]

  
    
#%%

# Plot the illiteracy rate versus fertility
_ = plt.plot(illiteracy, fertility, marker='.', linestyle='none')

# Set the margins and label axes
plt.margins(0.02)
_ = plt.xlabel('percent illiterate')
_ = plt.ylabel('fertility')

# Show the plot
plt.show()

# Show the Pearson correlation coefficient
print(pearson_r(illiteracy, fertility))


#%%

# Plot the illiteracy rate versus fertility
_ = plt.plot(illiteracy, fertility, marker='.', linestyle='none')
plt.margins(0.02)
_ = plt.xlabel('percent illiterate')
_ = plt.ylabel('fertility')

# Perform a linear regression using np.polyfit(): a, b
a, b = np.polyfit(illiteracy,fertility,1)

# Print the results to the screen
print('slope =', a, 'children per woman / percent illiterate')
print('intercept =', b, 'children per woman')

# Make theoretical line to plot
x = np.array([0,100])
y = a * x + b

# Add regression line to your plot
_ = plt.plot(x, y)

# Draw the plot
plt.show()
    
#%%


# Specify slopes to consider: a_vals
a_vals = np.linspace(0,0.1,200)

# Initialize sum of square of residuals: rss
rss = np.empty_like(a_vals)

# Compute sum of square of residuals for each value of a_vals
for i, a in enumerate(a_vals):
    rss[i] = np.sum((fertility - a * illiteracy - b)**2)

# Plot the RSS
plt.plot(a_vals, rss, '-')
plt.xlabel('slope (children per woman / percent illiterate)')
plt.ylabel('sum of square of residuals')

plt.show()

    
#%%

x=np.array([ 10.,   8.,  13.,   9.,  11.,  14.,   6.,   4.,  12.,   7.,   5.])
y=np.array([  8.04,   6.95,   7.58,   8.81,   8.33,   9.96,   7.24,   4.26,   10.84,   4.82,   5.68])
    
#%%


# Perform linear regression: a, b
a, b = np.polyfit(x,y,1)

# Print the slope and intercept
print(a, b)

# Generate theoretical x and y data: x_theor, y_theor
x_theor = np.array([3, 15])
y_theor = x_theor * a + b

# Plot the Anscombe data and theoretical line
_ = plt.plot(x,y,marker='.',linestyle='none')
_ = plt.plot(x_theor,y_theor)

# Label the axes
plt.xlabel('x')
plt.ylabel('y')

# Show the plot
plt.show()
    

#%%

anscombe_x = [np.array([ 10.,   8.,  13.,   9.,  11.,  14.,   6.,   4.,  12.,   7.,   5.]),
              np.array([ 10.,   8.,  13.,   9.,  11.,  14.,   6.,   4.,  12.,   7.,   5.]),
              np.array([ 10.,   8.,  13.,   9.,  11.,  14.,   6.,   4.,  12.,   7.,   5.]),
              np.array([  8.,   8.,   8.,   8.,   8.,   8.,   8.,  19.,   8.,   8.,   8.])]

anscombe_y=[np.array([  8.04,   6.95,   7.58,   8.81,   8.33,   9.96,   7.24,   4.26,
                   10.84,   4.82,   5.68]),
            np.array([ 9.14,  8.14,  8.74,  8.77,  9.26,  8.1 ,  6.13,  3.1 ,  9.13,
                   7.26,  4.74]),
            np.array([  7.46,   6.77,  12.74,   7.11,   7.81,   8.84,   6.08,   5.39,
                   8.15,   6.42,   5.73]),
            np.array([  6.58,   5.76,   7.71,   8.84,   8.47,   7.04,   5.25,  12.5 ,
                   5.56,   7.91,   6.89])]
    

#%%

# Iterate through x,y pairs
for x, y in zip(anscombe_x , anscombe_y ):
    # Compute the slope and intercept: a, b
    a, b = np.polyfit(x,y,1)

    # Print the result
    print('slope:', a, 'intercept:', b)


#%%


img = plt.imread("425px-Anscombes_quartet.png")    
plt.imshow(img)





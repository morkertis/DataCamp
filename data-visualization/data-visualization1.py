
import matplotlib.pyplot as plt
import numpy as np

#*********data****************


year=np.array([1970, 1971, 1972, 1973, 1974, 1975, 1976, 1977, 1978, 1979, 1980, 1981, 1982, 1983,
         1984 ,1985 ,1986 ,1987 ,1988 ,1989, 1990 ,1991 ,1992 ,1993, 1994, 1995, 1996, 1997,
         1998, 1999, 2000 ,2001 ,2002 ,2003 ,2004 ,2005 ,2006 ,2007 ,2008 ,2009 ,2010 ,2011])

physical_sciences=np.array([13.8,14.9,14.8,16.5,18.2,19.1,20.,21.3,22.5,23.7,24.6,25.7,27.3,27.6
                    ,28.,27.5,28.4,30.4,29.7,31.3,31.6,32.6,32.6,33.6,34.8,35.9,37.3,38.3
                    ,39.7,40.2,41.,42.2,41.1,41.7,42.1,41.6,40.8,40.7,40.7,40.7,40.2,40.1])

computer_science=np.array([13.6,13.6,14.9,16.4,18.9,19.8,23.9,25.7,28.1,30.2,32.5,34.8,36.3,37.1
                ,36.8,35.7,34.7,32.4,30.8,29.9,29.4,28.7,28.2,28.5,28.5,27.5,27.1,26.8
                ,27.,28.1,27.7,27.6,27.,25.1,22.2,20.6,18.6,17.6,17.8,18.1,17.6,18.2])


health=[77.1, 75.5, 76.9, 77.4, 77.9, 78.9, 79.2, 80.5, 81.9, 82.3, 83.5, 84.1, 84.4, 84.6, 85.1, 
        85.3, 85.7, 85.5, 85.2, 84.6, 83.9, 83.5, 83.0, 82.4, 81.8, 81.5, 81.3, 81.9, 82.1, 83.5, 
        83.5, 85.1, 85.8, 86.5, 86.5, 86.0, 85.9, 85.4, 85.2, 85.1, 85.0, 84.8]


education=[74.53532758, 74.14920369, 73.55451996, 73.50181443, 73.33681143, 72.80185448, 72.16652471, 72.45639481, 73.19282134, 
           73.82114234, 74.98103152, 75.84512345, 75.84364914, 75.95060123, 75.86911601, 75.92343971, 76.14301516, 76.96309168, 
           77.62766177, 78.11191872, 78.86685859, 78.99124597, 78.43518191, 77.26731199, 75.81493264, 75.12525621, 75.03519921, 
           75.16370129999999, 75.48616027, 75.83816206, 76.69214284, 77.37522931, 78.64424394, 78.54494815, 78.65074774, 79.06712173, 
           78.68630551, 78.72141311, 79.19632674, 79.53290870000001, 79.61862451, 79.43281184]



#%%
# Plot in blue the % of degrees awarded to women in the Physical Sciences
plt.plot(year, physical_sciences,color= 'blue')

# Plot in red the % of degrees awarded to women in Computer Science
plt.plot(year, computer_science,color=  'red')

# Display the plot
plt.show()

#%%



# Create plot axes for the first line plot
plt.axes([0.05,0.05,0.425,0.9])

# Plot in blue the % of degrees awarded to women in the Physical Sciences
plt.plot(year, physical_sciences, color='blue')

# Create plot axes for the second line plot
plt.axes([0.525,0.05,0.425,0.9])

# Plot in red the % of degrees awarded to women in Computer Science
plt.plot(year, computer_science, color='red')

# Display the plot
plt.show()

#%%

#subplot(nrows, ncols, index) *index start from 1
# Create a figure with 1x2 subplot and make the left subplot active
plt.subplot(1,2,1)

# Plot in blue the % of degrees awarded to women in the Physical Sciences
plt.plot(year, physical_sciences, color='blue')
plt.title('Physical Sciences')

# Make the right subplot active in the current 1x2 subplot grid
plt.subplot(1,2,2)

# Plot in red the % of degrees awarded to women in Computer Science
plt.plot(year, computer_science, color='red')
plt.title('Computer Science')

# Use plt.tight_layout() to improve the spacing between subplots
plt.tight_layout()
plt.show()



#%%

# Create a figure with 2x2 subplot layout and make the top left subplot active
plt.subplot(2,2,1)

# Plot in blue the % of degrees awarded to women in the Physical Sciences
plt.plot(year, physical_sciences, color='blue')
plt.title('Physical Sciences')

# Make the top right subplot active in the current 2x2 subplot grid 
plt.subplot(2,2,2)

# Plot in red the % of degrees awarded to women in Computer Science
plt.plot(year, computer_science, color='red')
plt.title('Computer Science')

# Make the bottom left subplot active in the current 2x2 subplot grid
plt.subplot(2,2,3)

# Plot in green the % of degrees awarded to women in Health Professions
plt.plot(year, health, color='green')
plt.title('Health Professions')

# Make the bottom right subplot active in the current 2x2 subplot grid
plt.subplot(2,2,4)

# Plot in yellow the % of degrees awarded to women in Education
plt.plot(year, education, color='yellow')
plt.title('Education')

# Improve the spacing between subplots and display them
plt.tight_layout()
plt.show()

#%%


# Plot the % of degrees awarded to women in Computer Science and the Physical Sciences
plt.plot(year,computer_science, color='red') 
plt.plot(year, physical_sciences, color='blue')

# Add the axis labels
plt.xlabel('Year')
plt.ylabel('Degrees awarded to women (%)')

# Set the x-axis range
plt.xlim(1990,2010)

# Set the y-axis range
plt.ylim(0,50)

# Add a title and display the plot
plt.title('Degrees awarded to women (1990-2010)\nComputer Science (red)\nPhysical Sciences (blue)')

# Save the image as 'xlim_and_ylim.png'
plt.savefig('xlim_and_ylim.png')

plt.show()


#%%



# Plot in blue the % of degrees awarded to women in Computer Science
plt.plot(year,computer_science, color='blue')

# Plot in red the % of degrees awarded to women in the Physical Sciences
plt.plot(year, physical_sciences,color='red')

# Set the x-axis and y-axis limits
plt.axis((1990,2010,0,50))

# Save the figure as 'axis_limits.png'
plt.savefig('axis_limits.png')

# Show the figure
plt.show()


#%%


# Specify the label 'Computer Science'
plt.plot(year, computer_science, color='red', label='Computer Science') 

# Specify the label 'Physical Sciences' 
plt.plot(year, physical_sciences, color='blue', label='Physical Sciences')

# Add a legend at the lower center
plt.legend(loc='lower center')

# Add axis labels and title
plt.xlabel('Year')
plt.ylabel('Enrollment (%)')
plt.title('Undergraduate enrollment of women')
plt.show()


#%%
# Plot with legend as before
plt.plot(year, computer_science, color='red', label='Computer Science') 
plt.plot(year, physical_sciences, color='blue', label='Physical Sciences')
plt.legend(loc='lower right')

# Compute the maximum enrollment of women in Computer Science: cs_max
cs_max = computer_science.max()

# Calculate the year in which there was maximum enrollment of women in Computer Science: yr_max
yr_max = year[computer_science.argmax()]

# Add a black arrow annotation
plt.annotate('Maximum',xy=(yr_max,cs_max),xytext=(yr_max+5,cs_max+5) ,arrowprops=dict(facecolor='black'))

# Add axis labels and title
plt.xlabel('Year')
plt.ylabel('Enrollment (%)')
plt.title('Undergraduate enrollment of women')
plt.show()


#%%


# Import matplotlib.pyplot
import matplotlib.pyplot as plt

# Set the style to 'ggplot'
plt.style.use('ggplot')

# Create a figure with 2x2 subplot layout
plt.subplot(2, 2, 1) 

# Plot the enrollment % of women in the Physical Sciences
plt.plot(year, physical_sciences, color='blue')
plt.title('Physical Sciences')

# Plot the enrollment % of women in Computer Science
plt.subplot(2, 2, 2)
plt.plot(year, computer_science, color='red')
plt.title('Computer Science')

# Add annotation
cs_max = computer_science.max()
yr_max = year[computer_science.argmax()]
plt.annotate('Maximum', xy=(yr_max, cs_max), xytext=(yr_max-1, cs_max-10), arrowprops=dict(facecolor='black'))

# Plot the enrollmment % of women in Health professions
plt.subplot(2, 2, 3)
plt.plot(year, health, color='green')
plt.title('Health Professions')

# Plot the enrollment % of women in Education
plt.subplot(2, 2, 4)
plt.plot(year, education, color='yellow')
plt.title('Education')

# Improve spacing between subplots and display them
plt.tight_layout()
plt.show()



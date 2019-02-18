# Import numpy and matplotlib.pyplot
import numpy as np
import matplotlib.pyplot as plt




# Generate two 1-D arrays: u, v
u = np.linspace(-2, 2, 41)
v = np.linspace(-1, 1, 21)

# Generate 2-D arrays from u and v: X, Y
X,Y = np.meshgrid(u,v)

# Compute Z based on X and Y
Z = np.sin(3*np.sqrt(X**2 + Y**2)) 

hp=[88, 193, 60, 98, 78, 100, 75, 76, 130, 140, 52, 88, 84, 148, 150, 130, 58, 82, 65, 110, 95, 110, 140, 170, 78, 90, 96, 95, 
    110, 75, 132, 150, 83, 85, 86, 75, 140, 139, 70, 52, 60, 84, 138, 180, 65, 67, 97, 150, 70, 100, 180, 129, 95, 90, 83, 75, 
    100, 85, 112, 67, 65, 88, 100, 75, 100, 70, 145, 110, 210, 80, 145, 69, 150, 198, 120, 92, 90, 115, 95, 75, 76, 67, 71, 115, 
    84, 91, 150, 215, 67, 175, 60, 175, 110, 95, 68, 150, 67, 95, 110, 105, 102, 110, 89, 66, 88, 75, 78, 105, 70, 103, 60, 150, 
    72, 170, 90, 110, 58, 152, 145, 139, 83, 69, 150, 67, 80, 71, 46, 105, 90, 110, 175, 80, 74, 150, 150, 65, 100, 48, 105, 90, 
    48, 105, 105, 88, 100, 75, 113, 190, 92, 80, 165, 180, 71, 97, 72, 105, 90, 75, 88, 155, 68, 90, 84, 87, 112, 87, 125, 108, 
    142, 97, 105, 75, 137, 150, 88, 145, 63, 95, 140, 88, 85, 70, 85, 115, 86, 79, 120, 120, 65, 110, 220, 115, 170, 100, 90, 225, 
    85, 65, 97, 90, 90, 49, 110, 70, 92, 53, 100, 190, 63, 90, 67, 65, 75, 100, 110, 60, 93, 88, 150, 100, 150, 88, 225, 68, 70, 
    208, 105, 74, 90, 110, 72, 97, 88, 88, 129, 85, 86, 150, 70, 48, 77, 65, 175, 90, 150, 110, 130, 53, 65, 158, 95, 61, 215, 
    100, 145, 68, 150, 88, 67, 105, 175, 160, 74, 135, 100, 67, 198, 180, 215, 100, 225, 155, 170, 81, 85, 95, 80, 92, 70, 149, 84, 
    97, 52, 72, 85, 52, 95, 71, 140, 100, 96, 150, 75, 107, 110, 75, 97, 133, 70, 67, 112, 145, 115, 98, 70, 78, 230, 63, 76, 105, 
    95, 62, 165, 165, 160, 190, 95, 180, 78, 120, 80, 75, 68, 67, 95, 140, 110, 72, 150, 95, 54, 153, 130, 170, 86, 97, 90, 145, 86, 
    79, 165, 83, 64, 92, 72, 140, 150, 96, 150, 80, 130, 100, 125, 90, 94, 76, 90, 150, 97, 85, 81, 78, 46, 84, 70, 153, 116, 100, 
    167, 88, 88, 88, 200, 125, 92, 110, 69, 67, 90, 150, 90, 71, 105, 62, 88, 122, 65, 88, 90, 68, 110, 88]

mpg=[18.0, 9.0, 36.1, 18.5, 34.3, 32.9, 32.2, 22.0, 15.0, 17.0, 44.0, 24.5, 32.0, 14.0, 15.0, 13.0, 36.0, 31.0, 32.0, 21.5, 
     19.0, 17.0, 16.0, 15.0, 23.0, 26.0, 32.0, 24.0, 21.0, 31.3, 32.7, 15.0, 23.0, 17.6, 28.0, 24.0, 14.0, 18.1, 36.0, 29.0, 
     35.1, 36.0, 16.5, 16.0, 29.9, 31.0, 27.2, 14.0, 32.1, 15.0, 12.0, 17.6, 25.0, 28.4, 29.0, 30.9, 20.0, 20.8, 22.0, 38.0, 
     31.0, 19.0, 16.0, 25.0, 22.0, 26.0, 13.0, 19.9, 11.0, 28.0, 15.5, 26.0, 14.0, 12.0, 24.2, 25.0, 22.5, 26.8, 23.0, 26.0, 
     30.7, 31.0, 27.2, 21.5, 29.0, 20.0, 13.0, 14.0, 38.0, 13.0, 24.5, 13.0, 25.0, 24.0, 34.1, 13.0, 44.6, 20.5, 18.0, 23.2, 
     20.0, 24.0, 25.5, 36.1, 23.0, 24.0, 18.0, 26.6, 32.0, 20.3, 27.0, 17.0, 21.0, 13.0, 24.0, 17.0, 39.1, 14.5, 13.0, 20.2, 
     27.0, 35.0, 15.0, 36.4, 30.0, 31.9, 26.0, 16.0, 20.0, 18.6, 14.0, 25.0, 33.0, 14.0, 18.5, 37.2, 18.0, 44.3, 18.0, 28.0, 
     43.4, 20.6, 19.2, 26.4, 18.0, 28.0, 26.0, 13.0, 25.8, 28.1, 13.0, 16.5, 31.5, 24.0, 15.0, 18.0, 33.5, 32.4, 27.0, 13.0, 
     31.0, 28.0, 27.2, 21.0, 19.0, 25.0, 23.0, 19.0, 15.5, 23.9, 22.0, 29.0, 14.0, 15.0, 27.0, 15.0, 30.5, 25.0, 17.5, 34.0, 
     38.0, 30.0, 19.8, 25.0, 21.0, 26.0, 16.5, 18.1, 46.6, 21.5, 14.0, 21.6, 15.5, 20.5, 23.9, 12.0, 20.2, 34.4, 23.0, 24.3, 
     19.0, 29.0, 23.5, 34.0, 37.0, 33.0, 18.0, 15.0, 34.7, 19.4, 32.0, 34.1, 33.7, 20.0, 15.0, 38.1, 26.0, 27.0, 16.0, 17.0, 
     13.0, 28.0, 14.0, 31.5, 34.5, 11.0, 16.0, 31.6, 19.1, 18.5, 15.0, 18.0, 35.0, 20.2, 13.0, 31.0, 22.0, 11.0, 33.5, 43.1, 
     25.4, 40.8, 14.0, 29.8, 16.0, 20.6, 18.0, 33.0, 31.8, 13.0, 20.0, 32.0, 13.0, 23.7, 19.2, 37.0, 18.0, 19.0, 32.3, 18.0, 
     13.0, 12.0, 36.0, 18.2, 19.0, 30.0, 15.0, 11.0, 10.0, 16.0, 14.0, 16.9, 13.0, 25.0, 21.0, 21.1, 26.0, 28.0, 29.0, 16.0, 
     26.6, 19.0, 32.8, 22.0, 19.0, 31.0, 23.0, 29.5, 17.5, 19.0, 24.0, 14.0, 28.0, 21.0, 22.4, 36.0, 18.0, 16.2, 39.4, 30.0, 
     18.0, 17.5, 28.8, 22.0, 34.2, 30.5, 16.0, 38.0, 41.5, 27.9, 22.0, 29.8, 17.7, 15.0, 14.0, 15.5, 17.5, 12.0, 29.0, 15.5, 
     35.7, 26.0, 30.0, 33.8, 18.0, 13.0, 20.0, 32.4, 16.0, 27.5, 23.0, 14.0, 17.0, 16.0, 23.0, 24.0, 27.0, 15.0, 27.0, 28.0, 
     14.0, 33.5, 39.0, 24.0, 26.5, 19.4, 15.0, 25.5, 14.0, 27.4, 13.0, 19.0, 17.0, 28.0, 22.0, 30.0, 18.0, 14.0, 22.0, 23.8, 
     24.0, 26.0, 26.0, 30.0, 29.0, 14.0, 25.4, 19.0, 12.0, 20.0, 27.0, 22.3, 10.0, 19.2, 26.0, 16.0, 37.3, 26.0, 20.2, 13.0, 
     21.0, 25.0, 20.5, 37.7, 36.0, 20.0, 37.0, 18.0, 27.0, 29.5, 17.5, 25.1]

#%%
# Display the resulting image with pcolor()
plt.pcolor(Z)


# Save the figure to 'sine_mesh.png'
plt.savefig('sine_mesh.png')


plt.show()

#%%


# Generate a default contour map of the array Z
plt.subplot(2,2,1)
plt.contour(X, Y, Z)
plt.title('default contour')

# Generate a contour map with 20 contours
plt.subplot(2,2,2)
plt.contour(X, Y, Z,20)
plt.title('20 contour')

# Generate a default filled contour map of the array Z
plt.subplot(2,2,3)
plt.contourf(X, Y, Z)
plt.title('default contour-f')


# Generate a default filled contour map with 20 contours
plt.subplot(2,2,4)
plt.contourf(X, Y, Z,20)
plt.title('20 contour-f')


# Improve the spacing between subplots
plt.tight_layout()

# Display the figure
plt.show()




#%%


# Create a filled contour plot with a color map of 'viridis'
plt.subplot(2,2,1)
plt.contourf(X,Y,Z,20, cmap='viridis')
plt.colorbar()
plt.title('Viridis')

# Create a filled contour plot with a color map of 'gray'
plt.subplot(2,2,2)
plt.contourf(X,Y,Z,20, cmap='gray')
plt.colorbar()
plt.title('Gray')

# Create a filled contour plot with a color map of 'autumn'
plt.subplot(2,2,3)
plt.contourf(X,Y,Z,20, cmap='autumn')
plt.colorbar()
plt.title('Autumn')

# Create a filled contour plot with a color map of 'winter'
plt.subplot(2,2,4)
plt.contourf(X,Y,Z,20, cmap='winter')
plt.colorbar()
plt.title('Winter')

# Improve the spacing between subplots and display them
plt.tight_layout()
plt.show()

#%%

# Generate a 1-D histogram
plt.subplot(2,2,1)
plt.hist(hp,bins=20,color='blue')
plt.title('hp')

plt.subplot(2,2,2)
plt.hist(mpg,bins=20)
plt.title('mpg')

plt.show()



#%%


# Generate a 2-D histogram
plt.hist2d(hp,mpg,bins=(20,20),range=((40, 235), (8, 48)))

# Add a color bar to the histogram
plt.colorbar()

# Add labels, title, and display the plot
plt.xlabel('Horse power [hp]')
plt.ylabel('Miles per gallon [mpg]')
plt.title('hist2d() plot')
plt.show()


#%%


# Generate a 2d histogram with hexagonal bins
plt.hexbin(hp,mpg,gridsize=(15,12),extent=(40, 235, 8, 48))

           
# Add a color bar to the histogram
plt.colorbar()

# Add labels, title, and display the plot
plt.xlabel('Horse power [hp]')
plt.ylabel('Miles per gallon [mpg]')
plt.title('hexbin() plot')
plt.show()


#%%


# Load the image into an array: img
img = plt.imread('480px-Astronaut-EVA.jpg')

# Print the shape of the image
print(img.shape)

# Display the image
plt.imshow(img)

# Hide the axes
plt.axis('off')
plt.show()

#%%

#TODO Colormap Possible values

'''
Accent, Accent_r, Blues, Blues_r, BrBG, BrBG_r, BuGn, BuGn_r, BuPu, BuPu_r, CMRmap, CMRmap_r, Dark2, Dark2_r, GnBu, GnBu_r, 
Greens, Greens_r, Greys, Greys_r, OrRd, OrRd_r, Oranges, Oranges_r, PRGn, PRGn_r, Paired, Paired_r, Pastel1, Pastel1_r, Pastel2, 
Pastel2_r, PiYG, PiYG_r, PuBu, PuBuGn, PuBuGn_r, PuBu_r, PuOr, PuOr_r, PuRd, PuRd_r, Purples, Purples_r, RdBu, RdBu_r, RdGy, 
RdGy_r, RdPu, RdPu_r, RdYlBu, RdYlBu_r, RdYlGn, RdYlGn_r, Reds, Reds_r, Set1, Set1_r, Set2, Set2_r, Set3, Set3_r, Spectral, Spectral_r, 
Wistia, Wistia_r, YlGn, YlGnBu, YlGnBu_r, YlGn_r, YlOrBr, YlOrBr_r, YlOrRd, YlOrRd_r, afmhot, afmhot_r, autumn, autumn_r, binary, 
binary_r, bone, bone_r, brg, brg_r, bwr, bwr_r, cool, cool_r, coolwarm, coolwarm_r, copper, copper_r, cubehelix, cubehelix_r, flag, 
flag_r, gist_earth, gist_earth_r, gist_gray, gist_gray_r, gist_heat, gist_heat_r, gist_ncar, gist_ncar_r, gist_rainbow, gist_rainbow_r, 
gist_stern, gist_stern_r, gist_yarg, gist_yarg_r, gnuplot, gnuplot2, gnuplot2_r, gnuplot_r, gray, gray_r, hot, hot_r, hsv, hsv_r, inferno, 
inferno_r, jet, jet_r, magma, magma_r, nipy_spectral, nipy_spectral_r, ocean, ocean_r, pink, pink_r, plasma, plasma_r, prism, prism_r, 
rainbow, rainbow_r, seismic, seismic_r, spectral, spectral_r, spring, spring_r, summer, summer_r, terrain, terrain_r, viridis, viridis_r, 
winter, winter_r
'''

#%%

# Load the image into an array: img
img = plt.imread('480px-Astronaut-EVA.jpg')

# Print the shape of the image
print(img.shape)

# Compute the sum of the red, green and blue channels: intensity
intensity = img.sum(axis=2)

# Print the shape of the intensity
print(intensity)

# Display the intensity with a colormap of 'gray'
plt.imshow(intensity,cmap='gray')

# Add a colorbar
plt.colorbar()

# Hide the axes and show the figure
plt.axis('off')
plt.show()


#%%


# Load the image into an array: img
img = plt.imread('480px-Astronaut-EVA.jpg')

# Specify the extent and aspect ratio of the top left subplot
plt.subplot(2,2,1)
plt.title('extent=(-1,1,-1,1),\naspect=0.5') 
plt.xticks([-1,0,1])
plt.yticks([-1,0,1])
plt.imshow(img, extent=(-1,1,-1,1), aspect=0.5)

# Specify the extent and aspect ratio of the top right subplot
plt.subplot(2,2,2)
plt.title('extent=(-1,1,-1,1),\naspect=1')
plt.xticks([-1,0,1])
plt.yticks([-1,0,1])
plt.imshow(img, extent=(-1,1,-1,1), aspect=1)

# Specify the extent and aspect ratio of the bottom left subplot
plt.subplot(2,2,3)
plt.title('extent=(-1,1,-1,1),\naspect=2')
plt.xticks([-1,0,1])
plt.yticks([-1,0,1])
plt.imshow(img, extent=(-1,1,-1,1), aspect=2)

# Specify the extent and aspect ratio of the bottom right subplot
plt.subplot(2,2,4)
plt.title('extent=(-2,2,-1,1),\naspect=2')
plt.xticks([-2,-1,0,1,2])
plt.yticks([-1,0,1])
plt.imshow(img, extent=(-2,2,-1,1), aspect=2)

# Improve spacing and display the figure
plt.tight_layout()
plt.show()



#%%


# Load the image into an array: image
image = plt.imread('640px-Unequalized_Hawkes_Bay_NZ.jpg')

# Extract minimum and maximum values from the image: pmin, pmax
pmin, pmax = image.min(), image.max()
print("The smallest & largest pixel intensities are %d & %d." % (pmin, pmax))

# Rescale the pixels: rescaled_image
rescaled_image = 256*(image - pmin) / (pmax - pmin)
print("The rescaled smallest & largest pixel intensities are %.1f & %.1f." % 
      (rescaled_image.min(), rescaled_image.max()))

# Display the original image in the top subplot
plt.subplot(2,1,1)
plt.title('original image')
plt.axis('off')
plt.imshow(image,cmap='gray')

# Display the rescaled image in the bottom subplot
plt.subplot(2,1,2)
plt.title('rescaled image')
plt.axis('off')
plt.imshow(rescaled_image,cmap='gray')

plt.show()



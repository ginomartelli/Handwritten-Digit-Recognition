# TODO: Import necessary libraries
import matplotlib.pyplot as plt
from sklearn import datasets
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from scipy.ndimage import sobel
import math
from sklearn.preprocessing import MinMaxScaler

digits = datasets.load_digits()
print(digits.DESCR) # Print dataset description
n=4 # Number of images to visualize
# Visualize some images
# TODO: Graph the first 4 images from the data base 
def plot_digits(data, n):
    '''Plots the first n digits'''
    fig = plt.figure(figsize=(10, 5))
    for i in range(n):
        plt.subplot(1, n, i + 1)
        plt.imshow(data[i].reshape((8, 8)), cmap='binary')
        plt.axis('off')
    plt.show()

plot_digits(digits.data, n)
# Display at least one random sample par class (some repetitions of class... oh well)
def plot_multi(data, y):
    '''Plots 16 digits'''
    nplots = 16
    nb_classes = len(np.unique(y))
    cur_class = 0
    fig = plt.figure(figsize=(15,15))
    for j in range(nplots):
        plt.subplot(4,4,j+1)
        to_display_idx = np.random.choice(np.where(y == cur_class)[0])
        plt.imshow(data[to_display_idx].reshape((8,8)), cmap='binary')
        plt.title(cur_class)
        plt.axis('off')
        cur_class = (cur_class + 1) % nb_classes
    plt.show()


plot_multi(digits.data, digits.target)

##########################################
## Data exploration and first analysis
##########################################

def get_statistics_text(targets):

    # TODO: Write your code here, returning at least the following useful infos:
    # * Label names
    # * Number of elements per class
    unique, counts = np.unique(targets, return_counts=True)
    label_counts = dict(zip(unique, counts))
    label_names = [str(i) for i in unique]
    stats_text = f"Label names: {label_names}\n"
    stats_text += "Number of elements per class:\n"
    for label, count in label_counts.items():
        stats_text += f"Class {label}: {count} elements\n"
    stats_text += f"Total number of elements: {len(targets)}\n"
    stats_text += f"Unique labels: {len(unique)}\n"
    # Plot the distribution of samples per class
    plt.figure(figsize=(8, 4))
    plt.bar(label_counts.keys(), label_counts.values(), tick_label=label_names)
    plt.xlabel("Class label")
    plt.ylabel("Number of samples")
    plt.title("Distribution of samples per class")
    plt.show()
    return stats_text

stats_text = get_statistics_text(digits.target)
print(stats_text)
# TODO: Call the previous function and generate graphs and prints for exploring and visualising the database



##########################################
## Start data preprocessing
##########################################

# Access the whole dataset as a matrix where each row is an individual (an image in our case) 
# and each column is a feature (a pixel intensity in our case)
## X = [
#  [Pixel1, Pixel2, ..., Pixel64],  # Image 1 as a row
#  [Pixel1, Pixel2, ..., Pixel64],  # Image 2 as a row
#  [Pixel1, Pixel2, ..., Pixel64],  # Image 3 as a row
#  [Pixel1, Pixel2, ..., Pixel64]   # Image 4 as a row
#]

# TODO: Create a feature matrix and a vector of labels
X = digits.data  # Feature matrix with shape (n_samples, n_features)
y = digits.target  # Labels vector with shape (n_samples,)

# Print dataset shape
print(f"Feature matrix shape: {X.shape}. Max value = {np.max(X)}, Min value = {np.min(X)}, Mean value = {np.mean(X)}")
print(f"Labels shape: {y.shape}")


# TODO: Normalize pixel values to range [0,1]
scaler = MinMaxScaler()
F = scaler.fit_transform(X)

# Print matrix shape
print(f"Feature matrix F shape: {F.shape}. Max value = {np.max(F)}, Min value = {np.min(F)}, Mean value = {np.mean(F)}")

##########################################
## Dimensionality reduction
##########################################


### just an example to test, for various number of PCs
sample_index = 0
original_image = F[sample_index].reshape(8, 8)  # Reshape back to 8×8 for visualization

# TODO: Using the specific sample above, iterate the following:
# * Generate a PCA model with a certain value of principal components
# * Compute the approximation of the sample with this PCA model
# * Reconstruct a 64 dimensional vector from the reduced dimensional PCA space
# * Reshape the resulting approximation as an 8x8 matrix
# * Quantify the error in the approximation
# Finally: plot the original image and the 15 approximation on a 4x4 subfigure
pca = PCA(2)  # Example with 2 principal components
X =pca.fit_transform(F)  # Fit PCA and transform the data
plt.figure(figsize=(10, 5))
for i in range(1, 16):
    pca = PCA(n_components=i)  # Create PCA model with i components
    X_reduced = pca.fit_transform(F)  # Fit and transform the data
    X_approx = pca.inverse_transform(X_reduced)  # Reconstruct the original space
    approx_image = X_approx[sample_index].reshape(8, 8)  # Reshape to 8x8

    # Calculate error (mean squared error)
    error = np.mean((original_image - approx_image) ** 2)

    plt.subplot(4, 4, i)
    plt.imshow(approx_image, cmap='binary')
    plt.title(f'PCs: {i}\nError: {error:.4f}')
    plt.axis('off')
plt.subplot(4, 4, 16)
plt.imshow(original_image, cmap='binary')
plt.title('Original Image')
plt.axis('off')
plt.tight_layout()
plt.show()



#### TODO: Expolore the explanined variance of PCA and plot 
# the explained variance ratio for different numbers of principal components
pca = PCA().fit(F)  # Fit PCA to the feature matrix
explained_variance = pca.explained_variance_ratio_
print(explained_variance)   
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(explained_variance) + 1), explained_variance, marker='o')
plt.title('Explained Variance Ratio by PCA Components')
plt.xlabel('Number of Principal Components')
plt.ylabel('Explained Variance Ratio')
plt.xticks(range(1, len(explained_variance) + 1))
plt.grid()
plt.show()
# Plot cumulative explained variance ratio for different numbers of principal components
cumulative_variance = np.cumsum(explained_variance)
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', color='green')
plt.title('Cumulative Explained Variance Ratio by PCA Components')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.xticks(range(1, len(cumulative_variance) + 1))
plt.grid()
plt.show()

# Create the visualization plot


### TODO: Display the whole database in 2D: 
# * Use PCA to reduce the dimensionality of the feature matrix to 2D
pca = PCA(n_components=2)  # Create PCA model with 2 components
F_reduced = pca.fit_transform(F)  # Fit PCA and transform the feature matrix
# * Plot the reduced feature matrix with different colors for each class
plt.figure(figsize=(10, 6))
plt.scatter(F_reduced[:, 0], F_reduced[:, 1], c=y, cmap='tab10', s=10)
plt.colorbar(label='Class Label')
plt.title('PCA Reduced Feature Matrix')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid()
plt.show()


### TODO: Create a 20 dimensional PCA-based feature matrix

# Reduce the dimensionality to 20 components
pca = PCA(n_components=20)  # Create PCA model with 20 components
F_pca = pca.fit_transform(F)  # Fit PCA and transform the feature matrix
# Reshape the reduced feature matrix to have 20 components


# Print reduced feature matrix shape
print(f"Feature matrix F_pca shape: {F_pca.shape}")


##########################################
## Feature engineering
##########################################
### # Function to extract zone-based features
###  Zone-Based Partitioning is a feature extraction method
### that helps break down an image into smaller meaningful regions to analyze specific patterns.
def extract_zone_features(images): 
    upper=np.zeros((1,24)) 
    mid=np.zeros((1,16)) 
    lower=np.zeros((1,24)) 
    new_img=[] 
    for img in images: 
        upper[0:24]=img[0:24] 
        mid[0:16]=img[24:40] 
        lower[0:24]=img[40:64] 
        new_img.append([upper.mean(),mid.mean(),lower.mean()]) 
    return np.array(new_img) 



F_zones = extract_zone_features(F) 


print(f"Feature matrix F_zones shape: {F_zones.shape}") 



###TODO: Get used to the Sobel filter by applying it to an image and displaying both the original image and the result of applying the Sobel filter side by side 

sobel_h = sobel(F, 0)
sobel_v = sobel(F,1) # horizontal gradient sobel_v = sobel(F, 1) # vertical gradient magnitude = np.sqrt(sobel_h2 + sobel_v2) 
moyenne_sobel=[] 
magnitude = np.sqrt(sobel_h**2 + sobel_v**2) 
for i in range(sobel_h.shape[0]): 
    moyenne_sobel.append(abs((sobel_h[i,:].mean() + sobel_v[i,:].mean()) / 2)) 

original_image = F[0].reshape(8, 8) 
sobel_h_img = sobel_h[0].reshape(8,8) 
sobel_v_img = sobel_v[0].reshape(8,8) 
magnitude_img = magnitude[0].reshape(8,8) 
plt.figure(figsize=(12,8)) 
plt.subplot(3,2,1) 
plt.imshow(original_image, cmap="binary") 
plt.axis("off") 

plt.subplot(3,2,2) 
plt.imshow(sobel_h_img, cmap="binary") 
plt.axis("off") 

plt.subplot(3,2,3) 
plt.imshow(sobel_v_img, cmap="binary") 
plt.axis("off") 

plt.subplot(3,2,4) 
plt.imshow(magnitude_img, cmap="binary") 
plt.axis("off") 

plt.show() 

###TODO: Compute the average edge intensity for each image and return it as an n by 1 array 

F_edges = np.array(moyenne_sobel) 

#Print feature shape after edge extraction 

print(f"Feature matrix F_edges shape: {F_edges.shape}") 

#connect all the features together 

###TODO: Concatenate PCA, zone-based, and edge features 
print(F_pca.shape, F_zones.shape, F_edges.shape)
F_final = np.concatenate((F_pca,F_zones,F_edges.reshape(1797,1)), axis=1)  

###TODO: Normalize final features 

scaler = MinMaxScaler() 
F_final = scaler.fit_transform(F_final) 

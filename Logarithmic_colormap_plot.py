import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as colors

############################### Constants

e = 1 * 1.60217663 * 10**(-19)
epsilon = 8.8541878128 * 10**(-12)
c = 3 * 10**(8)
hbar = 1.054571817 * 10**(-34)
m = 9.10938 * 10**(-31)

############################### Parameters

# Grid resolution
resolution = 300

# Range of distances
initial_distance =  0.8*10**(-7)
final_distance = 10**(-3)
distance = np.logspace(np.log10(initial_distance), np.log10(final_distance), resolution)

# Fixed frequency
w = 10**(10)

# Evolution time
t = np.linspace((-0.25 * np.pi), (3.25 * np.pi), 10**5)

# Range of squeezing
r_p = np.log(np.sqrt((2 * m * w * (max(distance)**2)) / hbar))
r_x = -r_p

squeezing = np.linspace(-abs(r_p), abs(r_p), resolution)

entropy_Darwin_max = np.zeros((len(distance), len(squeezing)))
entropy_Coulomb_max = np.zeros((len(distance), len(squeezing)))

def maximum_sympletic_eigenvalue(gd, gc, omega, r, t):
    return 1/4 * (-2 + np.sqrt(2) * np.sqrt(abs(
        (1 / ((-4 * gd * gc + w**2)**2)) * 
        np.exp(-4 * r) * (
            3 * (np.exp(8 * r) * gd**2 + gc**2) * w**2 + 
            np.exp(4 * r) * (
                -4 * gd * gc * (gd**2 - 6 * gd * gc + gc**2) + 
                (gd**2 - 8 * gd * gc + gc**2) * w**2 + 
                2 * w**4
            ) - 
            4 * (np.exp(4 * r) * gd + gc)**2 * w**2 * np.cos(
                2 *(np.pi/2)
            ) + 
            (np.exp(8 * r) * gd**2 * w**2 + gc**2 * w**2 + 
            np.exp(4 * r) * (
                4 * gd * gc * (gd + gc)**2 - (gd**2 + gc**2) * w**2
            )) * np.cos(
                4 * (np.pi/2)
            )
        )
    )))

def Darwin_max(t, w, distance, squeezing, entropy_Darwin_max):
    i = 0
    while i < len(distance):
        d = distance[i]
        gd = (3 * (e**2) * w) / (16 * np.pi * epsilon * m * d * (c**2))
        gc = (-1) * ((e**2)) / (4 * np.pi * epsilon * m * w * (d**3))
        w_eff = np.sqrt(-4 * gd * gc + w**2)
        t_eff = np.linspace(min(t) / w_eff, max(t) / w_eff, len(t))
        j = 0
        while j < len(squeezing):
            r = squeezing[j]
            f = maximum_sympletic_eigenvalue(gd, gc, w_eff, r, t_eff)
            if f == 0:
                f = 10**(-16)
            S_D = (-1) * f * np.log(abs(f)) + (1 + f) * np.log(abs(1 + f))
            entropy_Darwin_max[j][i] = S_D
            j += 1
        i += 1

def Coulomb_max(t, w, distance, squeezing, entropy_Coulomb_max):
    i = 0
    while i < len(distance):
        d = distance[i]
        gd = 0
        gc = (-1) * ((e**2)) / (4 * np.pi * epsilon * m * w * (d**3))
        w_eff = np.sqrt(-4 * gd * gc + w**2)
        t_eff = np.linspace(min(t) / w_eff, max(t) / w_eff, len(t))
        j = 0
        while j < len(squeezing):
            r = squeezing[j]
            f = maximum_sympletic_eigenvalue(gd, gc, w, r, t_eff)
            if f == 0:
                f = 10**(-16)
            S_C = (-1) * f * np.log(abs(f)) + (1 + f) * np.log(abs(1 + f))
            entropy_Coulomb_max[j][i] = S_C
            j += 1
        i += 1

Darwin_max(t, w, distance, squeezing, entropy_Darwin_max)
Coulomb_max(t, w, distance, squeezing, entropy_Coulomb_max)

x = distance
y = squeezing
z = np.abs(np.array(entropy_Darwin_max)) #- np.array(entropy_Coulomb_max))

#z = np.abs(np.array(entropy_Coulomb_max))

z = np.array(z).reshape(len(x), len(y))

#z[z == 0] = np.min(z[z != 0])

def interpolate_zeros(arr):
    """
    Interpolates zero values in a 2D array with the average of their neighboring values.
    """
    # Find indices of zero values
    zero_indices = np.argwhere(arr == 0)

    for idx in zero_indices:
        i, j = idx
        neighbors = []
        
        # Collect values of the neighboring elements
        if i > 0:
            neighbors.append(arr[i-1, j])  # Top
        if i < arr.shape[0] - 1:
            neighbors.append(arr[i+1, j])  # Bottom
        if j > 0:
            neighbors.append(arr[i, j-1])  # Left
        if j < arr.shape[1] - 1:
            neighbors.append(arr[i, j+1])  # Right

        # Compute the average of the neighbors
        if neighbors:
            arr[i, j] = np.mean(neighbors)
    
    return arr

#print(z)

plt.plot(x, np.log(np.sqrt((2 * m * w * (x**2)) / hbar)), alpha=1, c='cyan', label=r'$\xi_p$', linestyle='-', linewidth=2)
plt.plot(x, -np.log(np.sqrt((2 * m * w * (x**2)) / hbar)), alpha=1, c='cyan', label=r'$\xi_x$', linestyle='dashed', linewidth=2)

plt.plot(x, np.log( ( (4*(c**2))/(3*(w*x)**2) )**(1/4) ), alpha=1, c='lawngreen', label=r'$\xi_{dip}$', linestyle='dashdot', linewidth=2)

plt.legend()

plt.text(1.45*10**(-4), 2.1, "Darwin", color='red', rotation=-13.5, fontsize=12, verticalalignment='bottom', c = 'lawngreen')
plt.text(0.9*10**(-4), 2.5, "Coulomb", color='red', rotation=-13.5, fontsize=12, verticalalignment='top', c = 'lawngreen')

plt.imshow(z, origin='lower', extent=[min(x), max(x), min(y), max(y)], aspect=(np.log10(x[len(x)-1]) - np.log10(x[len(x)-2])) / (y[2] - y[1]), norm=colors.LogNorm(vmin=abs(z.min()), vmax=abs(z.max())))

pcm = plt.pcolor(x, y, z, norm=colors.LogNorm(vmin=abs(z.min()), vmax=abs(z.max())))
#pcm = plt.pcolor(x, y, z)
cbar = plt.colorbar(pcm)
cbar.set_label(r'$ S_{C+D}^{max} $', fontsize=22.5, labelpad=12.5)
cbar.ax.tick_params(labelsize=10)
plt.set_cmap('plasma')

plt.scatter([50 * 10**(-6)], [0.95 * 6.1635753533924875], c='cyan', marker='o')

plt.xscale('log')  # Set x-axis to log scale

plt.plot(figsize=(8, 6))
plt.rcParams["figure.dpi"] = 500

plt.ylabel(r'$\xi$', fontsize=22.5)
plt.xlabel(r'$d$ (m)', fontsize=22.5)
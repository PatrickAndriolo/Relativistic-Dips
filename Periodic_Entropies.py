import numpy as np
import matplotlib.pyplot as plt


############################# Define constants

# Electric charge
e = 1*1.60217663*10**(-19)

# Magnetic permeability of vacuum
epsilon =  8.8541878128*10**(-12)

# Speed of light
c = 3*10**(8)

# Planck constant
hbar = 1.054571817*10**(-34)

# Electron's mass
m = 9.10938*10**(-31)

############################# Used variables

# Distance between the traps
d = 50*10**(-6)

# Frequency 
w = 10**10

# Time range (usually taken to be some multiple of pi times the inverse power of the frequency)
# Use a good amount of points for the curves reach exactly 0 in each oscillation
t  = np.linspace((0*np.pi), (3.25*np.pi), 10**5)

# Bounds for squeezing (Eq's (23) and (24))
r_p = (np.log(np.sqrt( (2*m*w*(d**2))/(hbar) )))
r_x = - r_p

# Using squeezing parameter as 95% of the maximum allowed:
r = abs(r_p) * (0.95) 

print(r)

print('momentum bound =' + str(r_p))
print('squeezing bound =' + str(r_x))


# Sympletic eigenvalue \sigma^{(1)} (Eq. 34)
def sympletic_eigenvalue(gd, gc, omega, r, t):
    return 1/4 * (-2 + np.sqrt(2) * np.sqrt((
        (1 / ((-4 * gd * gc + w**2)**2)) * 
        np.exp(-4 * r) * (
            3 * (np.exp(8 * r) * gd**2 + gc**2) * w**2 + 
            np.exp(4 * r) * (
                -4 * gd * gc * (gd**2 - 6 * gd * gc + gc**2) + 
                (gd**2 - 8 * gd * gc + gc**2) * w**2 + 
                2 * w**4
            ) - 
            4 * (np.exp(4 * r) * gd + gc)**2 * w**2 * np.cos(
                2 * t * np.sqrt(-4 * gd * gc + w**2)
            ) + 
            (np.exp(8 * r) * gd**2 * w**2 + gc**2 * w**2 + 
            np.exp(4 * r) * (
                4 * gd * gc * (gd + gc)**2 - (gd**2 + gc**2) * w**2
            )) * np.cos(
                4 * t * np.sqrt(-4 * gd * gc + w**2)
            )
        )
    )))
                
                
                

def Darwin(t,w,m,r,d):
    
    # Define coupling constants (Eq. (10))
    gd = (3*(e**2)*w)/(16*np.pi*epsilon*m*d*(c**2))
    gc = (-1)*((e**2))/(4*np.pi*epsilon*m*w*(d**3))
    
    w_eff = np.sqrt(-4 * gd * gc + w**2)
    
    t_eff  = np.linspace(min(t) / w_eff, max(t) / w_eff, len(t))
    
    # Calculate the sympletic eigenvalue
    f = sympletic_eigenvalue(gd, gc, w_eff, r, t_eff)   


    # Python sometimes truncates f -> 0, which gives us warnings because f is 
    # the input for the log of the entanglement entropy. So we approximate
    # the f = 0 to f = 10^(-50).
    for k in range(len(f)):
        
        if f[k]==0:
            f[k] = 10**(-50)                                                  
    
    # Entanglement entropy:
    S_D = (-1)*f*np.log(abs(f))+ (1+f)*np.log(abs(1+f))
    
    return S_D


def Coulomb(t,w,m,r,d):
    
    # In this case, g_D = 0 because of the nonrelativistic limit
    gc = (-1)*((e**2))/(4*np.pi*epsilon*m*w*(d**3))
    
    gd = 0

    w_eff = np.sqrt(-4 * gd * gc + w**2)
    
    t_eff  = np.linspace(min(t)/w, max(t)/w_eff, len(t))

    # Calculate the sympletic eigenvalue
    f = sympletic_eigenvalue(gd, gc, w_eff, r, t_eff)  
    
    # Python sometimes truncates f -> 0, which gives us warnings because f is 
    # the input for the log of the entanglement entropy. So we approximate
    # the f = 0 to f = 10^(-50).
    for k in range(len(f)):
        
        if f[k]==0:
            f[k] = 10**(-50)                                                    
    
    # Entanglement entropy:
    S_C = (-1)*f*np.log(abs(f))+ (1+f)*np.log(abs(1+f))
    
    return S_C


# Defining these couplings here for estimating anti-squeezed position
gd = (3*(e**2)*w)/(16*np.pi*epsilon*m*d*(c**2))
print('gd = ' + str(gd * np.exp(r)))
gc = (-1)*((e**2))/(4*np.pi*epsilon*m*w*(d**3))
print('gc = ' + str(gc * np.exp(-r)))
w_eff = np.sqrt(-4 * gd * gc + w**2)
t_eff  = np.linspace( min(t)/w_eff, max(t)/w_eff, len(t))

# Print relevant quantities
print('Effective frequency = ' "%.1E" % w_eff)
#print('Unsqueezed Delta x = ' + str(  np.sqrt(hbar/(2*m*w_eff))  ))
#print('Unsqueezed Delta p = ' + str( np.sqrt((hbar*w_eff*m)/(2))  ))
#print('Squeezed Delta x = ' + str(  np.sqrt(hbar/(2*m*w_eff))  ))
#print('Squeezed Delta p = ' + str( np.sqrt((hbar*w_eff*m)/(2))  ))
#print('Darwin coupling = '"%.1E" % gd)
#print('Coulomb coupling = ' "%.1E" % gc)


print( str( (max(Darwin(t,w,m,r,d)) - max(Coulomb(t,w,m,r,d))) ) )



# Define plot size and resolution:
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
plt.rcParams["figure.dpi"] = 500

# Individual curves
ax.plot(w_eff*t_eff/(2*np.pi), Darwin(t,w,m,r,d), alpha = 0.9, c = 'blue', label = r'$S_{C+D}$', linestyle='-',linewidth=3)
ax.plot(w_eff*t_eff/(2*np.pi), Coulomb(t,w,m,r,d), alpha = 0.9, c = 'green', label = r'$S_{C}$', linestyle='dashed',linewidth=3)


#Difference between curves
#ax.plot(w_eff*t_eff/(2*np.pi), Darwin(t,w,m,0.975*abs(r_p),d) - Coulomb(t,w,m,0.975*abs(r_p),d), alpha = 0.8, c = 'green', label = r'$d = 5\mu m$', linestyle='-',linewidth=3)
#ax.plot(w_eff*t_eff/(2*np.pi), Darwin(t,w,m,0.95*abs(r_p),d) - Coulomb(t,w,m,0.95*abs(r_p),d), alpha = 0.9, c = 'blue', label = r'$d = 50\mu m$', linestyle='dashed',linewidth=3)
#ax.plot(w_eff*t_eff/(2*np.pi), Darwin(t,w,m,0.9*abs(r_p),d) - Coulomb(t,w,m,0.9*abs(r_p),d), alpha = 1, c = 'purple', label = r'$d = 500\mu m$', linestyle='dotted',linewidth=3)

# Title, size of labels, etc.
ax.set_xlabel(r'$\frac{\omega_{eff} t}{2\pi}$', fontsize="30") #+ "  " +  r'$(+1.5\cdot 10^{6})$')
ax.set_ylabel(r'$S(t)$   ' + r'$(10^{-9})$', fontsize="22.5")
#ax.set_ylim([-0.1* 10**(-9), 3.4*10**(-9)])
#plt.title('Entanglement Entropies') # + '  ' + r'$(\omega=10kHz, \xi = 3, d=5\mu m )$')
#ax.legend(fontsize="20")
#ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
          #ncol=2, fancybox=True, shadow=True)
ax.legend(loc='lower center', bbox_to_anchor=(0.85, 0), fontsize="20")
ax.xaxis.set_tick_params(labelsize=15)
ax.yaxis.set_tick_params(labelsize=15)
#ax.set_yscale('log')
plt.show()


# Delta x at (omega*t) / (pi/2)
print()
print('Delta x = ' + str(  np.sqrt(hbar/(2*m*w_eff)) ) +  ' , Delta x at omega*t at pi/2 = ' + str(  np.exp(r)*np.sqrt(hbar/(2*m*w_eff)) )  +  ', small d = ' + str(d))
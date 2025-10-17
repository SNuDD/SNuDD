"""Provides solar zenith angles due to earth's motion."""

import numpy as np
from scipy.integrate import odeint as ODEint

from snudd import config



# EARTH ORBIT PARAMETERS
AU         = 1.495978707e11            # Astronomic unit in m
e_earth    = 0.0167                    # Earth orbit eccentricity
th_eccl    = 23.44 * np.pi / 180.      # Earth's ecliptic angle in radians
days_in_yr = 365.25                    # Number of days in earth year



# Euler method for solving Newton's equation of motion

a             = 1.0                                # Normalize the semimajor axis to 1 AU
mu            = 4 * np.pi**2                       # Normalize grav pot mu to 4 pi^2 
T_earth       = 2*np.pi * np.sqrt(a**3/mu)         # Orbital period in years
r_peri, r_apo = a*(1.-e_earth), a*(1.+e_earth) # Perihelion and aphelion distance as a function of semimajor axis a and eccentricity e_earth
v_peri, v_apo = [np.sqrt(mu*(2./r - 1./a)) for r in (r_peri, r_apo)] # Perihelion and aphelion scalar velocities







# HELPER FUNCTIONS
############################


def deriv(X, t):
    """Returns the derivative of the position vector and velocity vector for the system of 1st order eqs"""
    x, v = X.reshape(2, -1)                # Setting the position vector to the current (2D) position and the velocity to the current velocity
    acc  = -x * mu * ((x**2).sum())**-1.5  # The acceleration is given by - mu r_vec/r^3 (But me normalize mu = G M = 1)
    return np.hstack((v, acc))             # Returning the velocity and acceleration


def azimuth(t):
    """Returns the azimuth in earth's frame as a function of time in years.
    Starts at azimuth=0 at t=0"""
    period = 23. + 56./60. # Earth's revolution period in hours
    return 2*np.pi / period * (t * days_in_yr * 24) 


def zenith(times, thetas, th_det):
    """Returns the solar neutrino zenith angle (taken from below the horizon) 
    for a given time series (in years) with corresponding true anomalies (thetas) at detector latitude (th_det)"""
    nu_dot_n = np.cos(thetas) * (np.cos(th_eccl)*np.sin(th_det)*np.cos(azimuth(times)) + np.sin(th_eccl)*np.cos(th_det)) + \
               np.sin(thetas) * np.sin(th_det)*np.sin(azimuth(times))
    
    return np.pi/2 - np.arccos(nu_dot_n)





# SOLAR ZENITH ANGLES
############################




class SolarAngles():
    """Below horizon zenith angles at detector latitude."""


    def __init__(self, latitude, t0, T):
        """Latitude in degrees."""
        self.lat  = (90. - latitude) * np.pi / 180.    # Latitude in radians
        self.t0   = t0                                 # Start of data taking period in days after perihelion (~ 3rd Jan)
        self.tdat = T                                  # Number of days of data taking from t0



    def orbit(self):
        """Calculate earht's orbit during data taking period"""

        # intial 2D boundary conditions at perihelion: x0, y0, vx0, vy0
        X0 = np.array([r_peri, 0, 0, v_peri]) 

        # Time steps
        index_start = 0
        times_dat   = np.linspace(self.t0, self.t0 + self.tdat, self.tdat*24*60)  # integration time series in minutes
        if self.t0 > 0:
            index_start = int(self.t0)*24 # Evolution time steps in hours till data taking start date
            times_init  = np.linspace(0, self.t0 , index_start)  # time evolution to t0 in hours
            times = np.concatenate((times_init, times_dat), axis=None)
        else:
            times = times_dat

        # SOLVE the differential equations with initial conditions, times in units of years
        coords, info = ODEint(deriv, X0, times/days_in_yr, full_output=True)

        # Return the time series and coordinates; remove first and last coordinate element from ODE solving to match length of times array
        return times[index_start:], coords[index_start+1:-1]
    


    def zenith_angles(self):
        """Get zenith angles at detector location over data taking period."""

        times, coords = self.orbit() 
        x, y   = coords.T[:2]                           # Earth's 2D orbit during data taking period
        thetas = np.arctan2(y, x)                       # True anomaly (angle around Sun taken from perihelion)
        dist   = np.sqrt(np.pow(x,2) + np.pow(y,2))     # Earth-Sun distance r in AU

        # Shifting the angels to the interval [0, 2 Pi] instead of [-Pi, Pi]
        for i in range(len(thetas)):
            if thetas[i] < 0: thetas[i] = thetas[i] + 2*np.pi

        # Calculate zenith angles in radians from below the horizon
        zens = zenith(times/days_in_yr, thetas, self.lat) 

        return times, zens


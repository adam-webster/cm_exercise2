'''
Symplectic euler time integration of 2 particles
interacting under a morse potential in 3D euclidean space.

Produces plots of the relative positions of the particles
and the energy, both as functions of time.
Also saves these to file.
'''

import sys
import math
import numpy as np
import matplotlib.pyplot as pyplot
from particle3D import Particle3D

def force_dw(p1, p2, a, D, r):
    """
    Method to return the morse force between 2 particles.
    Force on particle1 is given by
    F(p1, p2) = -2aD(1-exp[-a(mod(p1-p2)-r)])exp[-a(mod(p1-p2)-r)](p1-p2)

    :param p1: position of particle 1 in 3D euclidean space given by Numpy array
    :param p2: position of particle 2 in 3D euclidean space given by Numpy array
    :param a: parameter a system specific, read in from file
    :param D: parameter D system specific, read in from file
    :param r: parameter r system specific, read in from file
    :return: force acting on particle as Numpy array
    """
    exp = np.exp(-a*(np.absolute(p1.position-p2.position)- r))
    force = -2*a*D*(1-exp)*exp*np.linalg.norm(p1.position-p2.position)
    return force

def pot_energy_dw(p1, p2, a, D, r):
    """
    Method to return the morse potential energy
    of 2 particles at p1 and p2
    V(p1, p2) = D*((1-exp[-a(mod(p1-p2)-r)])^2 -1)

    :param p1: position of particle 1 in 3D euclidean space given by Numpy array
    :param p2: position of particle 2 in 3D euclidean space given by Numpy array
    :param a: parameter a system specific, read in from file
    :param D: parameter D system specific, read in from file
    :param r: parameter r system specific, read in from file
    :return: potential energy of pair as float
    """
    exp = np.exp(-a*(np.absolute(p1.position-p2.position)- r))
    potential = D*((1 -exp)**2 -1)
    return potential

# Begin main code
def main():

    # Read name of output file from command line
    if len(sys.argv)!=2:
        print("Wrong number of arguments.")
        print("Usage: " + sys.argv[0] + " <output file>")
        quit()
    else:
        outfile_name = sys.argv[1]

    # Open output file
    outfile = open(outfile_name, "w")

    # Set up simulation parameters
    dt = 0.00001
    numstep = 5000
    time = 0.0
    a = 2.65374
    D = 5.21322
    r = 1.20752

    # Set up particle initial conditions: needs to be changed to read in from file
    #particle1 = Particle3D(pos, vel, mass, label)
    p1 = Particle3D(np.array([2.0,0.0,0.0]), np.array([0.1,0.0,0.0]), 16.0, "Particle 1")
    p2 = Particle3D(np.array([2.0,0.0,0.0]), np.array([-0.1,0.0,0.0]), 16.0, "Particle 2")
    seperation = p1.position - p2.position
    # Write out initial conditions
    energy = p1.kinetic_energy() + p2.kinetic_energy() + pot_energy_dw(p1, p2, a, D, r)
    #outfile.write("{0:f} {4:12.8f}\n".format(time, p1.position[0], p1.position[1], p1.position[2], energy))
    outfile.write(str(time) + str(seperation) + str(energy))

    # Initialise data lists for plotting later
    time_list = [time]
    pos_list = [p1.position-p2.position]
    energy_list = [energy]

    # Start the time integration loop

    for i in range(numstep):
        # Update particle position
        p1.leap_pos1st(dt)
        p2.leap_pos1st(dt)

        # Calculate force
        force1 = force_dw(p1, p2, a, D, r)
        force2 = - force1
        # Update particle velocity
        p1.leap_velocity(dt, force1)
        p2.leap_velocity(dt, force2)
        # Increase time
        time = time + dt

        # Output particle information
        energy = p1.kinetic_energy() +p2.kinetic_energy() + pot_energy_dw(p1, p2, a, D, r)
        #outfile.write("{0:f} {1:f} {2:f} {3:f} {4:12.8f}\n".format(time, p1.position[0], p1.position[1], p1.position[2], energy))
        outfile.write(str(time) + str(seperation) + str(energy))
        # Append information to data lists
        time_list.append(time)
        pos_list.append(p1.position-p2.position)
        energy_list.append(energy)


    # Post-simulation:

    # Close output file
    outfile.close()

    # Plot particle trajectory to screen
    pyplot.title('Symplectic Euler: position vs time')
    pyplot.xlabel('Time')
    pyplot.ylabel('Position')
    pyplot.plot(time_list, pos_list)
    pyplot.show()

    # Plot particle energy to screen
    pyplot.title('Symplectic Euler: total energy vs time')
    pyplot.xlabel('Time')
    pyplot.ylabel('Energy')
    pyplot.plot(time_list, energy_list)
    pyplot.show()


# Execute main method:
main()

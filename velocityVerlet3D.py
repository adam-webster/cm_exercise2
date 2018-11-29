'''
Velocity Verlet time integration of 2 particles
interacting under a morse potential in 3D euclidean space.

Produces plots of the relative seperations of the particles
and the total energy, both as functions of time.
Also saves these to file.
'''

import sys
import math
import numpy as np
import matplotlib.pyplot as pyplot
from Particle3D import Particle3D

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
    exponent = math.exp(-a*(np.linalg.norm(p1.position-p2.position)- r))
    force = -2*a*D*(1-exponent)*exponent*((p1.position-p2.position)/np.linalg.norm(p1.position-p2.position))
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
    exponent = math.exp(-a*(np.linalg.norm(p1.position-p2.position)- r))
    potential = D*((1 -exponent)**2 -1)
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

    #read in initial conditions from appropriate files
    file_handle = open("N2_input.txt","r")
    p1 = Particle3D.from_file(file_handle)
    p2 = Particle3D.from_file(file_handle)
    settings = open("N2_potential_settings.txt", "r")
    a = float(settings.readline())
    D = float(settings.readline())
    r = float(settings.readline())

    #Set up simulation parameters
    dt = 0.105
    numstep = 105
    time = 0.0
    
    #Calculate seperation vector and absolute value of it
    vector_sep = Particle3D.vector_seperation(p1, p2)
    seperation = np.linalg.norm(vector_sep)

    # Write out initial conditions
    energy = p1.kinetic_energy() + p2.kinetic_energy() + pot_energy_dw(p1, p2, a, D, r)
    outfile.write("{0:f} {1:f} {2:f} {3:f} {4:12.8f}\n".format(time, vector_sep[0], vector_sep[1], vector_sep[2], energy))

    # Calculate force
    force1 = force_dw(p1, p2, a, D, r)
    force2 = -force1

    # Initialise data lists for plotting later
    time_list = [time]
    pos_list = [seperation]
    energy_list = [energy]

    # Start the time integration loop

    for i in range(numstep):
        # Update particle positions and seperation
        p1.leap_pos2nd(dt, force1)
        p2.leap_pos2nd(dt, force2)
        vector_sep = Particle3D.vector_seperation(p1, p2)
        seperation = np.linalg.norm(p1.position - p2.position)

        #update forces
        force1_new = force_dw(p1, p2, a, D, r)
        force2_new = -force1_new

        # Update particle velocity by averaging new and existing force
        p1.leap_velocity(dt, 0.5*(force1 + force1_new))
        p2.leap_velocity(dt, 0.5*(force2 + force2_new))

        #redefine force
        force1 = force1_new
        force2 = force2_new

        # Increase time
        time = time + dt

        # Output particle information
        energy = p1.kinetic_energy() + p2.kinetic_energy() + pot_energy_dw(p1, p2, a, D, r)
        outfile.write("{0:f} {1:f} {2:f} {3:f} {4:12.8f}\n".format(time, vector_sep[0], vector_sep[1], vector_sep[2], energy))

        # Append information to data lists
        time_list.append(time)
        pos_list.append(seperation)
        energy_list.append(energy)


    # Post-simulation:

    # Close output file
    outfile.close()

    # Plot particle trajectory to screen
    pyplot.title('Velocity Verlet: Seperation vs Time')
    pyplot.xlabel('Time, 10.18fs')
    pyplot.ylabel('Seperation, Angstroms')
    pyplot.plot(time_list, pos_list)
    pyplot.show()

    # Plot particle energy to screen
    pyplot.title('Velocity Verlet: Total Energy vs Time')
    pyplot.xlabel('Time, 10.18fs')
    pyplot.ylabel('Energy, eV')
    pyplot.plot(time_list, energy_list)
    pyplot.show()


# Execute main method:
main()


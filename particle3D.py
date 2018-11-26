"""
Particle3D, a class to describe 3D particles
"""

class Particle3D(object):
    """
    Class to describe 3D particles.

    Properties:
    position(numpy array) - position in 3D euclidean space
    velocity(numpy array) - velocity in 3D euclidean space
    mass(float) - particle mass
    label(string) - name for the particle

    Methods:
    * formatted output
    * kinetic energy
    * first-order velocity update
    * first- and second order position updates
    """

    def __init__(self, pos, vel, mass, label):
        """
        Initialise a Particle3D instance

        :param pos: position as numpy array
        :param vel: velocity as numpy array
        :param mass: mass as float
        :param label: particle label as a string
        """

        self.position = pos
        self.velocity = vel
        self.mass = mass
        self.label = label

    def __str__(self):
        """
        Define output format.
        For particle at (2.0, 0.5, 1.0) this will print as
        "<label> x = 2.0, y = 0.5, z = 1.0"
        """

        return str(self.label) + ", x = " + str(self.position[0]) + ", y = " + str(self.position[1]) + ", z = " + str(self.position[2])

    def kinetic_energy(self):
        """
        Return kinetic energy as
        1/2*mass*vel^2
        """

        return 0.5*self.mass*self.velocity**2

    # Time integration methods

    def leap_velocity(self, dt, force):
        """
        First-order velocity update,
        v(t+dt) = v(t) + dt*F(t)

        :param dt: timestep as float
        :param force: force on particle as float
        """

        self.velocity = self.velocity + dt*force/self.mass

    def leap_pos1st(self, dt):
        """
        First-order position update,
        x(t+dt) = x(t) + dt*v(t)

        :param dt: timestep as float
        """

        self.position = self.position + dt*self.velocity

    def leap_pos2nd(self, dt, force):
        """
        Second-order position update,
        x(t+dt) = x(t) + dt*v(t) + 1/2*dt^2*F(t)

        :param dt: timestep as float
        :param force: current force as float
        """

        self.position = self.position + dt*self.velocity + 0.5*dt**2*force/self.mass

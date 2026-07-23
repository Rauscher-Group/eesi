import numpy as np
import sys

def energy(theta,J):
    """
    Computes energy for a configuration, `theta` with energy parameter `J`
    """
    dtheta = np.diff(theta)
    dtheta = dtheta - 2.0*np.pi*np.rint(dtheta/(2.0*np.pi))
    return J*np.sum(np.cos(dtheta))

def mcxy(N=200,J=1,n_eq=1_000,n_prod=1_000_000,n_save=1_000):
    """
    Runs a simple Monte Carlo simulation of a 1D classical XY model.
    Returns 

    Inputs: 
        N [int]         :   Length of systems, i.e. number of spins
        J [float]       :   Dimensionless coupling parameter
        n_eq [int]      :   Number of equilibration steps
        n_prod [int]    :   Number of production steps
        n_save [int]    :   Number of configurations to save
    
    Returns:
        confs [np.ndarray]  :   Array of sampled configurations, 
                                shape (n_prod/n_save,N)

    """    

    # random initialization
    theta = 2.0*np.pi*np.random.uniform(size=N)
    ener = energy(theta,J)

    naccept = 0
    for _ in range(n_eq):
        # propose new move, change in angles is hard-coded for now
        theta_new = (theta + 0.1*np.random.uniform(size=N)) % (2.0*np.pi)
        ener_new  = energy(theta_new,J)
        
        # boltzmann factor of energy difference
        boltz = np.exp(ener - ener_new)
        if np.random.uniform() < boltz:
            theta = theta_new
            ener = ener_new
            naccept += 1

    print("Equilibration complete, {0:.1f}%  acceptance.".format(100.0*naccept/n_eq))
    
    naccept = 0
    #io = open(outfile,"w")
    eners = []
    confs = []
    save_every = n_prod // n_save
    for i in range(n_prod):
        # propose new move
        theta_new = (theta + 0.1*np.random.uniform(size=N)) % (2.0*np.pi)
        ener_new  = energy(theta_new,J)

        # boltzmann factor of energy difference
        boltz = np.exp(ener - ener_new)
        if np.random.uniform() < boltz:
            theta = theta_new
            ener = ener_new
            naccept += 1

        # save configurations as needed
        if i % save_every == 0:
            confs.append(theta)
            eners.append(ener)
            #out = ",".join(f"{val:.3f}" for val in theta)+"\n"
            #io.write(out)
    
    # write some output for sanity check
    print("Equilibration complete, {0:.1f}%  acceptance.".format(100.0*naccept/n_prod))
    print(f"Mean energy: {np.mean(eners)/N}")

    # return data
    return np.array(confs), np.array(eners)

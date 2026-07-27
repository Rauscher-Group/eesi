import numpy as np
import sys

def energy(theta,J):
    """
    Computes energy for a configuration, `theta` with energy parameter `J`
    """
    dtheta = np.diff(theta)
    dtheta = (dtheta + np.pi) % (2 * np.pi) - np.pi
    return -J*np.sum(np.cos(dtheta))

def sample_p1_exact(B, N, J, rng=None):
    """
    Draws exact i.i.d. samples from the open-chain Boltzmann distribution.

    `energy` depends on `theta` only through the bond variables
    `Delta_i = theta_{i+1} - theta_i`, and `(theta_1, Delta_1, ..., Delta_{N-1})`
    is a bijection of the torus with unit Jacobian, so *exactly*

        p1(theta) = Uniform(theta_1) * prod_i vonMises(Delta_i; mu=0, kappa=J)

    with the bonds mutually independent. Sampling is therefore O(BN), exact, and
    free of the equilibration and autocorrelation that `mcxy` carries; `mcxy` is
    retained as an independent check (see tests/test_xy_dataset.py).

    Inputs:
        B [int]                     :   Number of configurations to draw
        N [int]                     :   Length of system, i.e. number of spins
        J [float]                   :   Dimensionless coupling parameter
        rng [np.random.Generator]   :   Optional generator, for reproducibility

    Returns:
        confs [np.ndarray]  :   Array of sampled configurations wrapped onto
                                (-pi, pi], shape (B, N)
    """
    rng = np.random.default_rng() if rng is None else rng
    theta_1 = rng.uniform(-np.pi, np.pi, size=(B, 1))
    dtheta = rng.vonmises(mu=0.0, kappa=J, size=(B, N - 1))
    theta = np.concatenate([theta_1, theta_1 + np.cumsum(dtheta, axis=1)], axis=1)
    return (theta + np.pi) % (2 * np.pi) - np.pi

def mcxy(N=200,J=1,n_eq=1_000,n_prod=1_000_000,n_save=1_000):
    """
    Runs a simple Monte Carlo simulation of a 1D classical XY model.
    Returns

    Prefer `sample_p1_exact` for training data: it draws from the same
    distribution exactly, in O(BN), with no equilibration or autocorrelation.

    The proposal `theta + 0.1*U[0,1)^N` looks asymmetric but is not, in the
    variables that matter: it equals `theta + 0.05*1 + eps` with
    `eps ~ U(-0.05, 0.05)^N`, and the uniform shift leaves every bond
    `Delta_i` -- hence the energy -- unchanged. Detailed balance holds in the
    bonds; the absolute angles just pick up a deterministic +0.05 drift per
    accepted step, which wraps and leaves the `theta_1` marginal uniform.

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

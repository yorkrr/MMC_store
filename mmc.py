import numpy as np
from scipy.optimize import minimize
import vector
#这个是错的，不是真MMC，而是仅用theta共线假设，遍历所有phi，计算对应的p_T与实验测得的met最接近的就是
M_TAU = 1.777  # τ lepton mass in GeV

# Function to compute neutrino four-momentum given visible τ decay products and assumed neutrino direction
def vis_to_neutrino(vis, nu_phi, nu_theta):#这里为什么中微子动量方向假想好了，哦因为之后还能求重建的动量与缺失横动量的残差
    p_vis = np.sqrt(vis.px**2 + vis.py**2 + vis.pz**2)
    m_vis = np.sqrt(np.max(vis.E**2 - p_vis**2, 0))

    cos_theta = (vis.px * np.sin(nu_theta) * np.cos(nu_phi) +
                 vis.py * np.sin(nu_theta) * np.sin(nu_phi) +
                 vis.pz * np.cos(nu_theta)) / p_vis#cos_theta 表示 可见τ衰变产物的动量方向 与 假设的中微子方向 之间夹角的余弦值

    denominator = 2 * (vis.E - p_vis * cos_theta)
    p_nu_mag = np.where(abs(denominator) < 1e-6, 0, (M_TAU**2 - m_vis**2) / denominator)

    nu_px = p_nu_mag * np.sin(nu_theta) * np.cos(nu_phi)
    nu_py = p_nu_mag * np.sin(nu_theta) * np.sin(nu_phi)
    nu_pz = p_nu_mag * np.cos(nu_theta)
    E_nu = p_nu_mag  # neutrino mass is approximated as zero
    nu = np.array([E_nu, nu_px, nu_py, nu_pz])

    return nu

# Likelihood function to minimize
def likelihood(params, vis1, vis2, met):
    nu1_phi, nu2_phi = params
    nu1_theta = np.arccos(vis1.py / np.sqrt(vis1.px**2 + vis1.py**2 + vis1.pz**2))#nu_theta直接假设成和vis共线的？
    nu2_theta = np.arccos(vis2.py / np.sqrt(vis2.px**2 + vis2.py**2 + vis2.pz**2))

    nu1 = vis_to_neutrino(vis1, nu1_phi, nu1_theta)
    nu2 = vis_to_neutrino(vis2, nu2_phi, nu2_theta)

    met_residual_px = met.px - nu1[2] - nu2[1]
    met_residual_py = met.py - nu1[2] - nu2[1]
    return np.sum(met_residual_px**2 + met_residual_py**2)

# Missing Mass Calculator main function
def mmc_reconstruct(vis1, vis2, met, print_output=False):
    nu1f = []
    nu2f = []
    chi2 = []
    for i in range(len(vis1)):
        temp_vis1 = vis1[i]
        temp_vis2 = vis2[i]
        temp_met = met[i]
        
        result = minimize(likelihood, [0., 0.], args=(temp_vis1, temp_vis2, temp_met), bounds=[(-np.pi, np.pi), (-np.pi, np.pi)], method='L-BFGS-B')

        best_phi = result.x
        nu1_theta = np.arccos(temp_vis1.py / np.sqrt(temp_vis1.px**2 + temp_vis1.py**2 + temp_vis1.pz**2))
        nu2_theta = np.arccos(temp_vis2.py / np.sqrt(temp_vis2.px**2 + temp_vis2.py**2 + temp_vis2.pz**2))
        
        nu1 = vis_to_neutrino(temp_vis1, best_phi[0], nu1_theta)
        nu2 = vis_to_neutrino(temp_vis2, best_phi[1], nu2_theta)
        
        if result.success:
            chi2.append(1)
        else:
            chi2.append(0)
        if i == 0:
            nu1f = nu1
            nu2f = nu2
        else:
            nu1f = np.vstack((nu1f, nu1))
            nu2f = np.vstack((nu2f, nu2))
            
            
        # tau1 = vis1 + nu1
        # tau2 = vis2 + nu2

        # mass_squared = (tau1 + tau2).E**2 - (tau1 + tau2).px**2 + (tau1 + tau2).py**2 + (tau1 + tau2).pz**2
        # mass = np.where(mass_squared > 0, np.sqrt(mass_squared), 0)

        # if print_output:
        #     print(f"Reconstructed ττ mass: {mass:.2f} GeV")
    nu1 = vector.zip({
        "energy": nu1f[:,0],
        "px": nu1f[:,1],
        "py": nu1f[:,2],
        "pz": nu1f[:,3]
    })
    nu2 = vector.zip({
        "energy": nu2f[:,0],
        "px": nu2f[:,1],
        "py": nu2f[:,2],
        "pz": nu2f[:,3]
    })
    chi2 = np.array(chi2)
                
    return nu1, nu2, chi2

# Example event inputs
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import pickle
    raw_file_path = "/global/cfs/cdirs/m2616/avencast/Quantum_Entanglement/workspace_20250302_sig_apply/results/pi_pi/ml_export/pi_pi_recon_particles.pkl"
    with open(raw_file_path, "rb") as f:
        data = pickle.load(f)
    vis1_1 = data["tau_p_child1"][:100]
    vis1_2 = data["tau_p_child2"][:100]
    vis1_3 = data["tau_p_child3"][:100]
    vis2_1 = data["tau_m_child1"][:100]
    vis2_2 = data["tau_m_child2"][:100]
    vis2_3 = data["tau_m_child3"][:100]
    met = data["MET"][:100]
    truth_nu1 = data["truth_nu_p"][:100]
    truth_nu2 = data["truth_nu_m"][:100]
    vis1 = vis1_1 + vis1_2 + vis1_3
    vis2 = vis2_1 + vis2_2 + vis2_3
    nu1, nu2, chi2 = mmc_reconstruct(vis1, vis2, met, print_output=False)
    plt.hist(chi2, bins=2, histtype='step', label='χ2')
    plt.title("χ2 Distribution")
    plt.xlabel("χ2")
    plt.ylabel("Events")
    plt.legend()
    plt.savefig("chi2_distribution.png")
    plt.close()
    plt.hist(abs(truth_nu1.energy - nu1.energy)/(truth_nu1.energy + 1e-6).flatten(), bins=100, range=(0., 2.), histtype='step', label='Truth - Reco (τ+)')
    plt.title("Energy Resolution (τ+)")
    plt.xlabel("(Truth - Reco) / Truth")
    plt.ylabel("Events")
    plt.legend()
    plt.savefig("energy_resolution_tau_plus.png")
    plt.close()
    plt.hist(abs(truth_nu2.energy - nu2.energy)/(truth_nu2.energy + 1e-6).flatten(), bins=100, range=(0., 2.), histtype='step', label='Truth - Reco (τ-)')
    plt.title("Energy Resolution (τ-)")
    plt.xlabel("(Truth - Reco) / Truth")
    plt.ylabel("Events")
    plt.legend()
    plt.savefig("energy_resolution_tau_minus.png")
    plt.close()
    plt.hist(abs(truth_nu1.px - nu1.px)/(truth_nu1.px + 1e-6).flatten(), bins=100, range=(0., 2.), histtype='step', label='Truth - Reco (τ+)')
    plt.title("Momentum Resolution (τ+)")
    plt.xlabel("(Truth - Reco) / Truth")
    plt.ylabel("Events")
    plt.legend()
    plt.savefig("momentum_resolution_tau_plus.png")
    plt.close()
    plt.hist(abs(truth_nu2.px - nu2.px)/(truth_nu2.px + 1e-6).flatten(), bins=100, range=(0., 2.), histtype='step', label='Truth - Reco (τ-)')
    plt.title("Momentum Resolution (τ-)")
    plt.xlabel("(Truth - Reco) / Truth")
    plt.ylabel("Events")
    plt.legend()
    plt.savefig("momentum_resolution_tau_minus.png")
    plt.close()
    plt.hist(abs(truth_nu1.py - nu1.py)/(truth_nu1.py + 1e-6).flatten(), bins=100, range=(0., 2.), histtype='step', label='Truth - Reco (τ+)')
    plt.title("Momentum Resolution (τ+) py")
    plt.xlabel("(Truth - Reco) / Truth")
    plt.ylabel("Events")
    plt.legend()
    plt.savefig("momentum_resolution_tau_plus_py.png")
    plt.close()
    plt.hist(abs(truth_nu2.py - nu2.py)/(truth_nu2.py + 1e-6).flatten(), bins=100, range=(0., 2.), histtype='step', label='Truth - Reco (τ-)')
    plt.title("Momentum Resolution (τ-) py")
    plt.xlabel("(Truth - Reco) / Truth")
    plt.ylabel("Events")
    plt.legend()
    plt.savefig("momentum_resolution_tau_minus_py.png")
    plt.close()
    plt.hist(abs(truth_nu1.pz - nu1.pz)/(truth_nu1.pz + 1e-6).flatten(), bins=100, range=(0., 2.), histtype='step', label='Truth - Reco (τ+)')
    plt.title("Momentum Resolution (τ+) pz")
    plt.xlabel("(Truth - Reco) / Truth")
    plt.ylabel("Events")
    plt.legend()
    plt.savefig("momentum_resolution_tau_plus_pz.png")
    plt.close()
    plt.hist(abs(truth_nu2.pz - nu2.pz)/(truth_nu2.pz + 1e-6).flatten(), bins=100, range=(0., 2.), histtype='step', label='Truth - Reco (τ-)')
    
    # plt.hist(nu2.energy, bins=100, range=(0., 100.), histtype='step', label='Reco (τ-)')
    # plt.hist(truth_nu2.energy, bins=100, range=(0., 100.), histtype='step', label='Truth (τ-)')
    # plt.title("Energy Distribution (τ-)")
    # plt.xlabel("Energy (GeV)")
    # plt.ylabel("Events")
    # plt.legend()
    # plt.savefig("energy_distribution_tau_minus_fla.png")
    # plt.close()
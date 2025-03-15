# Let's try running the provided code snippet.
import numpy as np

# We'll try to import the vector package.
import vector

M_tau = 1.777

def compute_neutrino(v_vis, delta_phi):
    """
    根据可见τ衰变产物 v_vis（4矢量）以及中微子与其之间的方位角偏差 delta_phi，
    利用τ质量约束求出中微子四动量。
    假设中微子的极角与v_vis相同，即 theta_ν = theta_vis。
    """
    m_vis = v_vis.mass          # 可见粒子的质量
    E_vis = v_vis.E             # 能量
    p_vis = v_vis.mag           # 空间动量的模
    denom = 2 * (E_vis - p_vis * np.cos(delta_phi))
    if np.abs(denom) < 1e-6:
        return None  # 避免除0错误
    p_nu = (M_tau**2 - m_vis**2) / denom
    
    theta = v_vis.theta
    phi = v_vis.phi + delta_phi
    
    # 构造中微子4矢量（中微子视为无质量，所以E = |p|）
    nu = vector.obj(
        px = p_nu * np.sin(theta) * np.cos(phi),
        py = p_nu * np.sin(theta) * np.sin(phi),
        pz = p_nu * np.cos(theta),
        E  = p_nu
    )
    return nu

def reconstruct_neutrinos(v1, v2, met, n_steps=70, delta_phi_range=np.pi):
    """
    对两个τ衰变事件求解中微子4矢量，返回使MET匹配残差最小的解
    """
    met_x, met_y = met
    best_score = float('inf')
    best_nu1 = None
    best_nu2 = None
    best_dp1 = None
    best_dp2 = None

    delta_phis = np.linspace(-delta_phi_range, delta_phi_range, n_steps)
    for dphi1 in delta_phis:
        nu1 = compute_neutrino(v1, dphi1)
        if nu1 is None:
            continue
        for dphi2 in delta_phis:
            nu2 = compute_neutrino(v2, dphi2)
            if nu2 is None:
                continue
            sum_px = nu1.px + nu2.px
            sum_py = nu1.py + nu2.py
            score = (sum_px - met_x)**2 + (sum_py - met_y)**2
            if score < best_score:
                best_score = score
                best_nu1 = nu1
                best_nu2 = nu2
                best_dp1 = dphi1
                best_dp2 = dphi2
    return best_nu1, best_nu2, best_dp1, best_dp2, best_score

# -------------------------------
# 示例运行
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import pickle
    raw_file_path = "/global/cfs/cdirs/m2616/avencast/Quantum_Entanglement/workspace_20250302_sig_apply/results/pi_pi/ml_export/pi_pi_recon_particles.pkl"
    with open(raw_file_path, "rb") as f:
        data = pickle.load(f)
    vis1_1 = data["tau_p_child1"][:5000]
    vis1_2 = data["tau_p_child2"][:5000]
    vis1_3 = data["tau_p_child3"][:5000]
    vis2_1 = data["tau_m_child1"][:5000]
    vis2_2 = data["tau_m_child2"][:5000]
    vis2_3 = data["tau_m_child3"][:5000]
    met = data["MET"][:5000]
    truth_nu1 = data["truth_nu_p"][:5000]
    truth_nu2 = data["truth_nu_m"][:5000]
    vis1 = vis1_1 + vis1_2 + vis1_3
    vis2 = vis2_1 + vis2_2 + vis2_3
    nu1_arr = []
    nu2_arr = []
    for i in range(len(vis1)):
        v1 = vis1[i]
        v2 = vis2[i]
        tmet = [met[i].px, met[i].py]
        
        nu1, nu2, dp1, dp2, score = reconstruct_neutrinos(v1, v2, tmet)
        nu1_arr.append([nu1.E, nu1.px, nu1.py, nu1.pz])
        nu2_arr.append([nu2.E, nu2.px, nu2.py, nu2.pz])
    nu1_arr = np.array(nu1_arr)
    nu2_arr = np.array(nu2_arr)

    # Plot difference of px of nu1 between truth and reconstructed
    plt.hist((truth_nu1.px - nu1_arr[:,1])/(1e-6 + truth_nu1.px)*100., bins=50, range=(-200, 200))
    plt.xlabel("Δpx / px (%)")
    plt.ylabel("Counts")
    plt.title("Difference of px of nu1 between truth and reconstructed")
    plt.savefig("nu1_px_diff.png")
    plt.close()
    # Then, plot difference of py of nu1 between truth and reconstructed
    plt.hist((truth_nu1.py - nu1_arr[:,2])/(1e-6 + truth_nu1.py)*100., bins=50, range=(-200, 200))
    plt.xlabel("Δpy / py (%)")
    plt.ylabel("Counts")
    plt.title("Difference of py of nu1 between truth and reconstructed")
    plt.savefig("nu1_py_diff.png")
    plt.close()
    # Next, plot difference of pz of nu1 between truth and reconstructed
    plt.hist((truth_nu1.pz - nu1_arr[:,3])/(1e-6 + truth_nu1.pz)*100., bins=50, range=(-200, 200))
    plt.xlabel("Δpz / pz (%)")
    plt.ylabel("Counts")
    plt.title("Difference of pz of nu1 between truth and reconstructed")
    plt.savefig("nu1_pz_diff.png")
    plt.close()
    # Plot difference of px of nu2 between truth and reconstructed
    plt.hist((truth_nu2.px - nu2_arr[:,1])/(1e-6 + truth_nu2.px)*100., bins=50, range=(-200, 200))
    plt.xlabel("Δpx / px (%)")
    plt.ylabel("Counts")
    plt.title("Difference of px of nu2 between truth and reconstructed")
    plt.savefig("nu2_px_diff.png")
    plt.close()
    # Then, plot difference of py of nu2 between truth and reconstructed
    plt.hist((truth_nu2.py - nu2_arr[:,2])/(1e-6 + truth_nu2.py)*100., bins=50, range=(-200, 200))
    plt.xlabel("Δpy / py (%)")
    plt.ylabel("Counts")
    plt.title("Difference of py of nu2 between truth and reconstructed")
    plt.savefig("nu2_py_diff.png")
    plt.close()
    # Next, plot difference of pz of nu2 between truth and reconstructed
    plt.hist((truth_nu2.pz - nu2_arr[:,3])/(1e-6 + truth_nu2.pz)*100., bins=50, range=(-200, 200))
    plt.xlabel("Δpz / pz (%)")
    plt.ylabel("Counts")
    plt.title("Difference of pz of nu2 between truth and reconstructed")
    plt.savefig("nu2_pz_diff.png")
    plt.close()
    # Plot the distribution of nu1 px
    plt.hist(nu1_arr[:,1], bins=50)
    plt.xlabel("px")
    plt.ylabel("Counts")
    plt.title("Distribution of nu1 px")
    plt.savefig("nu1_px.png")
    plt.close()
    # Then, plot the distribution of nu1 py
    plt.hist(nu1_arr[:,2], bins=50)
    plt.xlabel("py")
    plt.ylabel("Counts")
    plt.title("Distribution of nu1 py")
    plt.savefig("nu1_py.png")
    plt.close()
    # Next, plot the distribution of nu1 pz
    plt.hist(nu1_arr[:,3], bins=50)
    plt.xlabel("pz")
    plt.ylabel("Counts")
    plt.title("Distribution of nu1 pz")
    plt.savefig("nu1_pz.png")
    plt.close()
        
    

# End of code.


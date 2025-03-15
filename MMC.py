import numpy as np
import vector
from scipy.stats import levy_stable
from scipy.optimize import fsolve

M_tau = 1.777

def P_deltaR(delta_R, p_tau, is_tau_1):
    npy = np.load("/global/homes/b/baihong/workdir/MMC/histogram.npz", allow_pickle=True)
    h1 = npy["h1"]
    h2 = npy["h2"]
    x1 = npy["x1"]
    y1 = npy["y1"]
    x2 = npy["x2"]
    y2 = npy["y2"]
    
    def get_density(xi, yi, H, xedges, yedges):
        ix = np.searchsorted(xedges, xi) - 1
        iy = np.searchsorted(yedges, yi) - 1
        # 边界判断，如果不在范围内则返回 0
        if ix < 0 or ix >= H.shape[0] or iy < 0 or iy >= H.shape[1]:
            return 0.0
        # 计算该 bin 的面积
        bin_area = (xedges[ix+1] - xedges[ix]) * (yedges[iy+1] - yedges[iy])
        # 返回该 bin 的概率值
        return H[ix, iy] * bin_area
    
    if is_tau_1:
        return get_density(p_tau, delta_R, h1, x1, y1)
    else:
        return get_density(p_tau, delta_R, h2, x2, y2) 

def get_p_mis(m_vis, p_vis, E_vis, theta_vis, phi_vis, theta_mis, phi_mis):
    """
    根据文章 Eq. (3) 对于 hadronic 衰变（m_mis = 0），τ 质量约束为：
    
      M_tau^2 = m_vis^2 + 2 [ E_vis * p_mis - p_vis * p_mis * cos(Δθ_vm) ]
    
    其中 cos(Δθ_vm) = sinθ_vis*sinθ_mis*cos(φ_vis - φ_mis) + cosθ_vis*cosθ_mis.
    故精确定义中微子动量：
    
      p_mis = (M_tau^2 - m_vis^2) / (2*(E_vis - p_vis*cos(Δθ_vm)))
    
    此处没有额外近似，严格按照文章公式实现。
    """
    cos_dtheta = np.sin(theta_vis) * np.sin(theta_mis) * np.cos(phi_vis - phi_mis) + \
                 np.cos(theta_vis) * np.cos(theta_mis)
    numerator = M_tau**2 - m_vis**2
    denominator = 2 * (E_vis - p_vis * cos_dtheta)
    # 若分母接近零，则返回 None（表示该网格点无解）
    if np.abs(denominator) < 1e-6:
        return None
    return numerator / denominator

def missing_ET_equations(thetas, phi_mis1, phi_mis2, Etx, Ety, vis_params1, vis_params2):
    """
    对于每个 τ 衰变，未知数为中微子极角 θ_mis。
    由横向 E/T 约束（文章 Eq. (3) 中的 E/T_x 和 E/T_y 方程）有：
    
       F1 = p_mis1*sin(θ_mis1)*cos(φ_mis1) + p_mis2*sin(θ_mis2)*cos(φ_mis2) - E/T_x = 0
       F2 = p_mis1*sin(θ_mis1)*sin(φ_mis1) + p_mis2*sin(θ_mis2)*sin(φ_mis2) - E/T_y = 0
    
    其中 p_mis1,2 分别由 τ 质量约束精确定义（见 get_p_mis）。
    该方程组不作任何简化，直接用于 fsolve 求解。
    """
    theta_mis1, theta_mis2 = thetas
    p_vis1, m_vis1, theta_vis1, phi_vis1 = vis_params1
    p_vis2, m_vis2, theta_vis2, phi_vis2 = vis_params2
    E_vis1 = np.sqrt(p_vis1**2 + m_vis1**2)
    E_vis2 = np.sqrt(p_vis2**2 + m_vis2**2)
    
    p_mis1 = get_p_mis(m_vis1, p_vis1, E_vis1, theta_vis1, phi_vis1, theta_mis1, phi_mis1)
    p_mis2 = get_p_mis(m_vis2, p_vis2, E_vis2, theta_vis2, phi_vis2, theta_mis2, phi_mis2)
    if p_mis1 is None or p_mis2 is None:
        return [1e6, 1e6]
    ETx_calc = p_mis1 * np.sin(theta_mis1) * np.cos(phi_mis1) + \
               p_mis2 * np.sin(theta_mis2) * np.cos(phi_mis2)
    ETy_calc = p_mis1 * np.sin(theta_mis1) * np.sin(phi_mis1) + \
               p_mis2 * np.sin(theta_mis2) * np.sin(phi_mis2)
    return [ETx_calc - Etx, ETy_calc - Ety]

def compute_neutrino_four_momenta(phi_mis1, phi_mis2, Etx, Ety, vis_params1, vis_params2):
    """
    对于给定的中微子方位角 (φ_mis1, φ_mis2) 及已知的 E/T_x, E/T_y，
    利用 fsolve 精确求解中微子极角 θ_mis1, θ_mis2，使得横向 E/T 约束满足。
    解得 p_mis1, p_mis2 后，构造中微子四动量（中微子质量设为 0）。
    该步骤严格依据文章描述，无任何为简化问题的近似。
    """
    theta_guess = [vis_params1[2], vis_params2[2]]
    args = (phi_mis1, phi_mis2, Etx, Ety, vis_params1, vis_params2)
    sol, infodict, ier, mesg = fsolve(missing_ET_equations, theta_guess, args=args, full_output=True)
    if ier != 1:
        return None
    theta_mis1, theta_mis2 = sol
    p_vis1, m_vis1, theta_vis1, phi_vis1 = vis_params1
    p_vis2, m_vis2, theta_vis2, phi_vis2 = vis_params2
    E_vis1 = np.sqrt(p_vis1**2 + m_vis1**2)
    E_vis2 = np.sqrt(p_vis2**2 + m_vis2**2)
    p_mis1 = get_p_mis(m_vis1, p_vis1, E_vis1, theta_vis1, phi_vis1, theta_mis1, phi_mis1)
    p_mis2 = get_p_mis(m_vis2, p_vis2, E_vis2, theta_vis2, phi_vis2, theta_mis2, phi_mis2)
    if p_mis1 is None or p_mis2 is None:
        return None
    nu1 = {
        'E': p_mis1,
        'px': p_mis1 * np.sin(theta_mis1) * np.cos(phi_mis1),
        'py': p_mis1 * np.sin(theta_mis1) * np.sin(phi_mis1),
        'pz': p_mis1 * np.cos(theta_mis1)
    }
    nu2 = {
        'E': p_mis2,
        'px': p_mis2 * np.sin(theta_mis2) * np.cos(phi_mis2),
        'py': p_mis2 * np.sin(theta_mis2) * np.sin(phi_mis2),
        'pz': p_mis2 * np.cos(theta_mis2)
    }
    return nu1, nu2, (theta_mis1, theta_mis2), (p_mis1, p_mis2)

def four_vector_sum(v1, v2):
    """返回两个四矢量的和（严格按照狭义相对论定义）"""
    return {
        'E': v1['E'] + v2['E'],
        'px': v1['px'] + v2['px'],
        'py': v1['py'] + v2['py'],
        'pz': v1['pz'] + v2['pz']
    }

def invariant_mass(v):
    """计算四矢量 v 的不变量质量"""
    mass2 = v['E']**2 - v['px']**2 - v['py']**2 - v['pz']**2
    return np.sqrt(max(mass2, 0))

def get_pseudorapidity(theta):
    """根据极角 θ 计算伪快度 η (η = -ln(tan(θ/2)))"""
    return -np.log(np.tan(theta / 2.0))

def event_likelihood(delta_R1, p_tau1, delta_R2, p_tau2):
    """
    定义事件似然函数：
    
       L = - log [ P(ΔR₁, pτ₁) × P(ΔR₂, pτ₂) ]
    
    这里没有对似然函数做任何简化，完全按照文章 Eq. (4) 实现。
    """
    P1 = P_deltaR(delta_R1, p_tau1, is_tau_1=True)
    P2 = P_deltaR(delta_R2, p_tau2, is_tau_1=False)
    if P1 * P2 <= 0:
        return np.inf
    return -np.log(P1 * P2)

def MMC_reconstruction(Etx, Ety, vis_params1, vis_params2, phi_grid_points=50):
    best_weight = -np.inf
    best_solution = None
    best_MtauTau = None
    results = []
    
    phi_vals = np.linspace(0, 2*np.pi, phi_grid_points, endpoint=False)
    for phi_mis1 in phi_vals:
        for phi_mis2 in phi_vals:
            sol = compute_neutrino_four_momenta(phi_mis1, phi_mis2, Etx, Ety, vis_params1, vis_params2)
            if sol is None:
                continue
            nu1, nu2, (theta_mis1, theta_mis2), (p_mis1, p_mis2) = sol
            p_vis1, m_vis1, theta_vis1, phi_vis1 = vis_params1
            p_vis2, m_vis2, theta_vis2, phi_vis2 = vis_params2
            E_vis1 = np.sqrt(p_vis1**2 + m_vis1**2)
            E_vis2 = np.sqrt(p_vis2**2 + m_vis2**2)
            vis1 = {
                'E': E_vis1,
                'px': p_vis1 * np.sin(theta_vis1) * np.cos(phi_vis1),
                'py': p_vis1 * np.sin(theta_vis1) * np.sin(phi_vis1),
                'pz': p_vis1 * np.cos(theta_vis1)
            }
            vis2 = {
                'E': E_vis2,
                'px': p_vis2 * np.sin(theta_vis2) * np.cos(phi_vis2),
                'py': p_vis2 * np.sin(theta_vis2) * np.sin(phi_vis2),
                'pz': p_vis2 * np.cos(theta_vis2)
            }
            tau1 = four_vector_sum(vis1, nu1)
            tau2 = four_vector_sum(vis2, nu2)
            MtauTau = invariant_mass(four_vector_sum(tau1, tau2))
            p_tau1 = np.sqrt(tau1['px']**2 + tau1['py']**2 + tau1['pz']**2)
            p_tau2 = np.sqrt(tau2['px']**2 + tau2['py']**2 + tau2['pz']**2)
            
            eta_vis1 = get_pseudorapidity(theta_vis1)
            eta_vis2 = get_pseudorapidity(theta_vis2)
            eta_mis1 = get_pseudorapidity(theta_mis1)
            eta_mis2 = get_pseudorapidity(theta_mis2)
            dphi1 = np.abs(phi_vis1 - phi_mis1)
            if dphi1 > np.pi:
                dphi1 = 2*np.pi - dphi1
            dphi2 = np.abs(phi_vis2 - phi_mis2)
            if dphi2 > np.pi:
                dphi2 = 2*np.pi - dphi2
            delta_R1 = np.sqrt((eta_vis1 - eta_mis1)**2 + dphi1**2)
            delta_R2 = np.sqrt((eta_vis2 - eta_mis2)**2 + dphi2**2)
            
            L_event = event_likelihood(delta_R1, p_tau1, delta_R2, p_tau2)
            weight = np.exp(-L_event)
            results.append((phi_mis1, phi_mis2, MtauTau, weight, nu1, nu2))
            if weight > best_weight:
                best_weight = weight
                best_solution = (phi_mis1, phi_mis2, nu1, nu2, theta_mis1, theta_mis2, p_mis1, p_mis2)
                best_MtauTau = MtauTau
    return best_solution, best_MtauTau, results

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
        Etx = met[i].px
        Ety = met[i].py
        vis_params1 = (v1.pt*np.cosh(v1.eta), v1.m, v1.theta, v1.phi)
        vis_params2 = (v2.pt*np.cosh(v2.eta), v2.m, v2.theta, v2.phi)
        
        best_solution, best_MtauTau, results = MMC_reconstruction(Etx, Ety, vis_params1, vis_params2, phi_grid_points=30)
        if best_solution is None:
            phi_mis1, phi_mis2, nu1, nu2, theta_mis1, theta_mis2, p_mis1, p_mis2 = best_solution
            nu1_arr.append([nu1['E'], nu1['px'], nu1['py'], nu1['pz']])
            nu2_arr.append([nu2['E'], nu2['px'], nu2['py'], nu2['pz']])
        else:
            nu1_arr.append([0, 0, 0, 0])
            nu2_arr.append([0, 0, 0, 0])
    nu1_arr = np.array(nu1_arr).reshape(-1, 4)
    nu2_arr = np.array(nu2_arr).reshape(-1, 4)

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
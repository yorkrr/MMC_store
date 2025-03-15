import numpy as np
from scipy.optimize import minimize
from vector import MomentumObject4D

# 辅助函数：根据τ质量约束计算中微子纵向动量
def solve_pz(vis, phi_mis, pt_mis, tau_mass=1.777):
    if pt_mis == 0:
        return 0.0
    p_vis = np.sqrt(vis.px**2 + vis.py**2 + vis.pz**2)
    E_vis = vis.energy
    p_mis = pt_mis / np.sin(phi_mis) if np.sin(phi_mis) != 0 else 0
    if p_mis == 0:
        return 0.0
    
    # 根据τ质量方程解算cosθ
    numerator = tau_mass**2 - vis.mass**2 - 2 * E_vis * np.sqrt(p_mis**2)
    denominator = 2 * p_vis * p_mis
    if denominator == 0:
        return 0.0
    cos_theta = numerator / denominator
    cos_theta = np.clip(cos_theta, -1, 1)
    theta = np.arccos(cos_theta)
    return p_mis * np.cos(theta)

def calculate_neutrino_momenta(
    vis1: MomentumObject4D,  # 可见产物1（如π+）的四动量
    vis2: MomentumObject4D,  # 可见产物2（如π-）的四动量
    met_x: float,            # 横向缺失能量x分量 (GeV)
    met_y: float,            # 横向缺失能量y分量 (GeV)
    tau_mass: float = 1.777  # τ轻子质量 (GeV/c²)
) -> tuple[MomentumObject4D, MomentumObject4D]:
    
    # 定义ΔR的概率密度函数（根据论文参数化）
    def delta_r_probability(delta_r, p_tau):
        # 示例参数化（需根据论文图2调整）
        sigma_gauss = 0.1 + 0.02 * p_tau  # 高斯分量
        sigma_landau = 0.2 + 0.03 * p_tau # 朗道分量
        gauss = np.exp(-0.5 * (delta_r / sigma_gauss)**2)
        landau = np.exp(-0.5 * (delta_r + sigma_landau)/sigma_landau)
        return 0.7*gauss + 0.3*landau  # 按论文混合比例

    # 定义目标函数：最小化负对数似然
    def objective(params):
        phi1, phi2 = params
        
        # 计算中微子横向动量（考虑MET约束）
        pt_mis1 = (met_x - (vis1.px + vis2.px)) / (np.cos(phi1) + 1e-10)
        pt_mis2 = (met_y - (vis1.py + vis2.py)) / (np.sin(phi2) + 1e-10)
        
        # 计算纵向动量
        pz_mis1 = solve_pz(vis1, phi1, pt_mis1, tau_mass)
        pz_mis2 = solve_pz(vis2, phi2, pt_mis2, tau_mass)
        
        # 构建中微子四动量
        nu1 = MomentumObject4D.from_xyzt(
            pt_mis1 * np.cos(phi1),
            pt_mis1 * np.sin(phi1),
            pz_mis1,
            np.hypot(pt_mis1, pz_mis1)
        )
        nu2 = MomentumObject4D.from_xyzt(
            pt_mis2 * np.cos(phi2),
            pt_mis2 * np.sin(phi2),
            pz_mis2,
            np.hypot(pt_mis2, pz_mis2)
        )
        
        # 计算ΔR并评估概率
        delta_r1 = np.sqrt((vis1.eta - nu1.eta)**2 + (vis1.phi - nu1.phi)**2)
        delta_r2 = np.sqrt((vis2.eta - nu2.eta)**2 + (vis2.phi - nu2.phi)**2)
        
        p_tau1 = np.sqrt(vis1.px**2 + vis1.py**2 + vis1.pz**2)
        p_tau2 = np.sqrt(vis2.px**2 + vis2.py**2 + vis2.pz**2)
        
        likelihood = (
            delta_r_probability(delta_r1, p_tau1) *
            delta_r_probability(delta_r2, p_tau2)
        )
        return -np.log(likelihood + 1e-10)

    # 使用优化器寻找最佳phi
    initial_guess = [vis1.phi + np.pi/4, vis2.phi - np.pi/4]  # 初始猜测
    result = minimize(objective, initial_guess, bounds=[(0, 2*np.pi), (0, 2*np.pi)])
    phi1_opt, phi2_opt = result.x
    
    # 计算最终动量
    pt_mis1 = (met_x - (vis1.px + vis2.px)) / np.cos(phi1_opt)
    pt_mis2 = (met_y - (vis1.py + vis2.py)) / np.sin(phi2_opt)
    
    pz_mis1 = solve_pz(vis1, phi1_opt, pt_mis1, tau_mass)
    pz_mis2 = solve_pz(vis2, phi2_opt, pt_mis2, tau_mass)
    
    nu1 = MomentumObject4D.from_xyzt(
        pt_mis1 * np.cos(phi1_opt),
        pt_mis1 * np.sin(phi1_opt),
        pz_mis1,
        np.sqrt(pt_mis1**2 + pz_mis1**2)
    )
    
    nu2 = MomentumObject4D.from_xyzt(
        pt_mis2 * np.cos(phi2_opt),
        pt_mis2 * np.sin(phi2_opt),
        pz_mis2,
        np.sqrt(pt_mis2**2 + pz_mis2**2)
    )
    
    return nu1, nu2

# 示例用法
if __name__ == "__main__":
    # 输入可见产物（假设τ→π+ν和τ→π-ν）
    pion_mass = 0.13957  # GeV/c²
    vis1 = MomentumObject4D.from_xyzt(
        10, 5, 3, 
        np.sqrt(10**2 + 5**2 + 3**2 + pion_mass**2)  # π+四动量
    )
    vis2 = MomentumObject4D.from_xyzt(
        -8, -6, 2,
        np.sqrt(8**2 + 6**2 + 2**2 + pion_mass**2)   # π-四动量
    )
    met_x = -2.0  # 示例MET（需根据实际情况调整）
    met_y = 1.0
    
    nu1, nu2 = calculate_neutrino_momenta(vis1, vis2, met_x, met_y)
    print(f"Neutrino 1: px={nu1.px:.2f}, py={nu1.py:.2f}, pz={nu1.pz:.2f}, E={nu1.energy:.2f}")
    print(f"Neutrino 2: px={nu2.px:.2f}, py={nu2.py:.2f}, pz={nu2.pz:.2f}, E={nu2.energy:.2f}")
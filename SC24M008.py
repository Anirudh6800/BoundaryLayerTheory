import numpy as np
import matplotlib.pyplot as plt

def falkner_skan_rhs(y, m):
    """
    Computes the right-hand side of the Falkner–Skan ODE.
    
    Parameters:
        y : array_like
            State vector [f, f', f''].
        m : float
            Pressure gradient parameter.
            
    Returns:
        dydeta : numpy array
            Array [f', f'', f'''].
    """
    return np.array([y[1],
                     y[2],
                     -((m + 1) / 2) * y[0] * y[2] - m * (1 - y[1]**2)])

def falkner_skan_rk4(m, fpp0, eta_max, h, Bf):
    """
    Uses the RK4 method to integrate the Falkner–Skan ODE from 0 to eta_max,
    and returns the error (i.e. f'(eta_max) - 1).
    
    Parameters:
        m : float
            Pressure gradient parameter.
        fpp0 : float
            Initial guess for f''(0).
        eta_max : float
            Maximum value of the similarity variable.
        h : float
            Step size.
        Bf : float
            Suction/blowing parameter.
    
    Returns:
        error : float
            f'(eta_max) - 1.
    """
    y = np.array([-Bf/(m+1), 0, fpp0], dtype=float)
    for _ in np.arange(0, eta_max, h):
        k1 = h * falkner_skan_rhs(y, m)
        k2 = h * falkner_skan_rhs(y + k1/2, m)
        k3 = h * falkner_skan_rhs(y + k2/2, m)
        k4 = h * falkner_skan_rhs(y + k3, m)
        y = y + (k1 + 2*k2 + 2*k3 + k4)/6.0
    error = y[1] - 1.0  # f'(eta_max) should equal 1
    return error

def falkner_skan_rk4_solution(m, fpp0, eta_max, h, Bf):
    """
    Solves the Falkner–Skan ODE using the RK4 method.
    
    Parameters:
        m : float
            Pressure gradient parameter.
        fpp0 : float
            Initial guess for f''(0).
        eta_max : float
            Maximum value of the similarity variable.
        h : float
            Step size.
        Bf : float
            Suction/blowing parameter.
    
    Returns:
        eta_vals : numpy array
            Array of eta values.
        f_vals : numpy array
            f(eta) values.
        f_prime_vals : numpy array
            f'(eta) values.
        f_double_prime_vals : numpy array
            f''(eta) values.
    """
    y = np.array([-Bf/(m+1), 0, fpp0], dtype=float)
    eta_vals = np.arange(0, eta_max+h, h)
    n = len(eta_vals)
    f_vals = np.zeros(n)
    f_prime_vals = np.zeros(n)
    f_double_prime_vals = np.zeros(n)
    
    for i in range(n):
        f_vals[i] = y[0]
        f_prime_vals[i] = y[1]
        f_double_prime_vals[i] = y[2]
        
        k1 = h * falkner_skan_rhs(y, m)
        k2 = h * falkner_skan_rhs(y + k1/2, m)
        k3 = h * falkner_skan_rhs(y + k2/2, m)
        k4 = h * falkner_skan_rhs(y + k3, m)
        y = y + (k1 + 2*k2 + 2*k3 + k4)/6.0
        
    return eta_vals, f_vals, f_prime_vals, f_double_prime_vals

def main():
    # Given parameters
    m_values = [2.0, 1.0, 0.6, 0.3, 0, -0.05, -0.08, -0.09043]
    tol = 1e-7
    max_iter = 300
    h = 0.01
    Bf = 0.0

    # Initial guesses for f''(0)
    s_guesses = [1.65, 1.2, 1.1, 1.0, 0.5, 0.3, 0.15, 0.1]

    # Prepare storage for results
    results = []

    # Prepare figures for plotting
    fig1 = plt.figure(1)
    plt.title("Velocity Profiles (f' vs. η)")
    plt.xlabel("η")
    plt.ylabel("f'(η)")
    plt.grid(True)
    
    fig2 = plt.figure(2)
    plt.title("Dimensionless Normal Velocity (v_dimless)")
    plt.xlabel("η")
    plt.ylabel("v_dimless")
    plt.grid(True)
    
    fig3 = plt.figure(3)
    plt.title("Non-Dimensional Shear Stress (f'')")
    plt.xlabel("η")
    plt.ylabel("f''(η)")
    plt.grid(True)
    
    # Get a colormap with as many colors as m values
    colors = plt.cm.jet(np.linspace(0, 1, len(m_values)))

    # Loop over each m value
    for i, m in enumerate(m_values):
        s0 = s_guesses[i]
        s1 = s0 + 0.05
        eta_max = 9.0
        
        # Secant method to determine the correct f''(0)
        for _ in range(max_iter):
            error0 = falkner_skan_rk4(m, s0, eta_max, h, Bf)
            error1 = falkner_skan_rk4(m, s1, eta_max, h, Bf)
            # Prevent division by zero
            if np.abs(error1 - error0) < 1e-12:
                break
            s_new = s1 - error1 * (s1 - s0) / (error1 - error0)
            if np.abs(s_new - s1) < tol:
                s1 = s_new
                break
            s0, s1 = s1, s_new

        # Solve the Falkner–Skan equation with the converged f''(0)
        eta_vals, f_vals, f_prime_vals, f_double_prime_vals = falkner_skan_rk4_solution(m, s1, eta_max, h, Bf)
        
        # Compute displacement thickness (δ*) and momentum thickness (θ)
        delta_star = np.trapz(1 - f_prime_vals, eta_vals)
        theta = np.trapz(f_prime_vals * (1 - f_prime_vals), eta_vals)
        H = delta_star/theta if theta != 0 else np.nan  # Shape factor
        cf_2 = f_double_prime_vals[0]  # friction coefficient (cf/2)
        lam = theta**2 * m          # Lambda parameter (avoid using "lambda" as variable)
        T_val = theta * cf_2        # T parameter
        F_theta = 2 * (T_val - (H + 2) * lam)  # F_theta parameter

        # Store results: m, δ*, θ, H, cf/2, lambda, T, F_theta
        results.append((m, delta_star, theta, H, cf_2, lam, T_val, F_theta))

        # Plot Velocity Profiles (f')
        plt.figure(1)
        plt.plot(eta_vals, f_prime_vals, color=colors[i], linewidth=1.5, label=f"m = {m:.2f}")
        # Identify inflection points (where f'' changes sign)
        inflection_indices = np.where(np.diff(np.sign(f_double_prime_vals)) != 0)[0]
        if inflection_indices.size > 0:
            plt.scatter(eta_vals[inflection_indices], f_prime_vals[inflection_indices], color=colors[i])
        
        # Plot Dimensionless Normal Velocity: v_dimless = 0.5 * η * (1 - f')
        v_dimless = 0.5 * eta_vals * (1 - f_prime_vals)
        plt.figure(2)
        plt.plot(eta_vals, v_dimless, color=colors[i], linewidth=1.5, label=f"m = {m:.2f}")
        
        # Plot Non-Dimensional Shear Stress (f'')
        plt.figure(3)
        plt.plot(eta_vals, f_double_prime_vals, color=colors[i], linewidth=1.5, label=f"m = {m:.2f}")
        
        # Also print a summary for each m value
        print(f"\nResults for m = {m:.2f} (f''(0) = {s1:.6f}):")
        print(f"δ*  = {delta_star:.4f}, θ = {theta:.4f}, H = {H:.4f}")
        print(f"cf/2 = {cf_2:.4f}, λ = {lam:.4f}, T = {T_val:.4f}, Fθ = {F_theta:.4f}")
    
    # Finalize plots
    plt.figure(1)
    plt.xlabel("η")
    plt.ylabel("f'(η)")
    plt.legend()
    
    plt.figure(2)
    plt.xlabel("η")
    plt.ylabel("v_dimless")
    plt.legend()
    
    plt.figure(3)
    plt.xlabel("η")
    plt.ylabel("f''(η)")
    plt.legend()
    
    # Print the summary table
    header = f"\n{'m':>8} {'δ*/δFS':>10} {'θ/δFS':>10} {'H':>10} {'cf/2':>10} {'λ':>10} {'T':>10} {'Fθ':>10}"
    print(header)
    print("-" * len(header))
    for res in results:
        print(f"{res[0]:8.2f} {res[1]:10.4f} {res[2]:10.4f} {res[3]:10.4f} {res[4]:10.4f} {res[5]:10.4f} {res[6]:10.4f} {res[7]:10.4f}")
    
    plt.show()



    results_array = np.array(results)  # Convert results to a NumPy array for indexing

    plt.subplot(1, 3, 1)
    plt.plot(results_array[:, 5], results_array[:, 3], 'o-')  # λ vs H
    plt.xlabel("λ")
    plt.ylabel("H")
    plt.title("λ vs H")
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.plot(results_array[:, 5], results_array[:, 6], 'o-')  # λ vs T
    plt.xlabel("λ")
    plt.ylabel("T")
    plt.title("λ vs T")
    plt.grid(True)

    plt.subplot(1, 3, 3)
    plt.plot(results_array[:, 5], results_array[:, 7], 'o-')  # λ vs Fθ
    plt.xlabel("λ")
    plt.ylabel("Fθ")
    plt.title("λ vs Fθ")
    plt.grid(True)

    plt.tight_layout()
    plt.show()




# The code above implements the Falkner–Skan boundary layer problem using the RK4 method for numerical integration.
if __name__ == '__main__':
    main()



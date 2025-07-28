import numpy as np

def generate_dataset(n=500, d=6, seed=None):
    if d != 6:
        raise ValueError("d must be 6")
    rng = np.random.default_rng(seed)
    z1, z2 = rng.normal(size=n), rng.normal(size=n)
    X = np.empty((n, d))
    X[:, :3] = z1[:, None] + rng.normal(size=(n, 3)) / 5
    X[:, 3:] = z2[:, None] + rng.normal(size=(n, 3)) / 5
    y1 = 3 * z1 + 2 * rng.normal(size=n)
    y2 = 3 * z1 - 1.5 * z2 + 2 * rng.normal(size=n)
    return X, y1, y2

def soft_threshold(x, lam):
    return np.sign(x) * np.maximum(np.abs(x) - lam, 0.0)

def centralize(X, y):
    xm, ym = X.mean(0), y.mean()
    Xc, yc = X - xm, y - ym
    sc = Xc.std(0, ddof=0)
    return Xc / sc, yc, xm, ym, sc

def lasso_cd(Xs, yc, lam, max_iter=1000, tol=1e-6):
    n, d = Xs.shape
    beta = np.zeros(d)
    norm2 = (Xs ** 2).sum(0)
    for _ in range(max_iter):
        beta_old = beta.copy()
        for j in range(d):
            r_j = yc - Xs @ beta + Xs[:, j] * beta[j]
            rho = Xs[:, j] @ r_j
            beta[j] = soft_threshold(rho / n, lam) / (norm2[j] / n)
        if np.max(np.abs(beta - beta_old)) < tol:
            break
    return beta

def lasso_with_intercept(X, y, lam):
    Xs, yc, xm, ym, sc = centralize(X, y)
    beta_std = lasso_cd(Xs, yc, lam)
    beta = beta_std / sc
    intercept = ym - xm @ beta
    return intercept, beta

if __name__ == "__main__":
    X, y_case1, y_case2 = generate_dataset(seed=42)

    lam_max = np.max(np.abs(X.T @ (y_case1 - y_case1.mean()))) / X.shape[0]
    lams = np.linspace(lam_max, lam_max * 0.1, 10)

    for lam in lams:
        b0_1, beta1 = lasso_with_intercept(X, y_case1, lam)
        b0_2, beta2 = lasso_with_intercept(X, y_case2, lam)
        print(f"\nλ = {lam:.4f}")
        print("Case 1  β0:", f"{b0_1:.4f}", "β:", np.round(beta1, 4))
        print("Case 2  β0:", f"{b0_2:.4f}", "β:", np.round(beta2, 4))

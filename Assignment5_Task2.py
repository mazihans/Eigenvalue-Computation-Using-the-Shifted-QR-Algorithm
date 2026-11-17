import numpy as np

def hessenberg(A):

    n = A.shape[0]
    H = A.copy()

    for k in range(n - 2):
        # Create the Householder vector
        x = H[k + 1:n, k]
        x_norm = np.linalg.norm(x)
        if x_norm == 0:
            continue

        e = np.zeros_like(x)
        e[0] = -x_norm * (1 if x[0] >= 0 else -1)
        u = x - e
        v = u / np.linalg.norm(u)

        # Apply the Householder transformation
        H[k + 1:n, k:n] -= 2 * np.outer(v, v @ H[k + 1:n, k:n])
        H[0:n, k + 1:n] -= 2 * np.outer(H[0:n, k + 1:n] @ v, v)

    return H

def Rayleigh_shift_deflation_QR(A, tol=1e-8, max_iter=1000):

    n = A.shape[0]
    H = hessenberg(A)
    eigenvalues = []
    total_iterations = 0

    for i in range(n):
        m = H.shape[0]
        if m == 1:
            eigenvalues.append(H[0, 0])
            break

        iter_count = 0
        while iter_count < max_iter:
            # Compute the Rayleigh shift
            mu = H[m - 1, m - 1]
            # Shifted QR decomposition
            Q, R = np.linalg.qr(H - mu * np.eye(m))
            H = R @ Q + mu * np.eye(m)
            iter_count += 1
            total_iterations += 1

            # Check for convergence
            if np.abs(H[m - 1, m - 2]) < tol:
                eigenvalues.append(H[m - 1, m - 1])
                H = H[:-1, :-1]  # Deflate the matrix
                break

        if iter_count == max_iter:
            raise ValueError("QR algorithm did not converge within the maximum number of iterations.")

    return np.array(eigenvalues), total_iterations

def random_symmetric_matrix(n):
    A = np.random.rand(n, n)
    return (A + A.T) / 2  # Make the matrix symmetric

def compare_eigenvalues(A, tol=1e-8):
    import numpy as np

    # Compute eigenvalues using the Rayleigh shift and deflation method
    eigenvalues_custom, _ = Rayleigh_shift_deflation_QR(A, tol=tol)

    # Compute eigenvalues using NumPy's built-in function
    eigenvalues_numpy = np.linalg.eigvals(A)

    # Sort both sets of eigenvalues for comparison
    eigenvalues_custom.sort()
    eigenvalues_numpy.sort()

    # Compare the two sets of eigenvalues
    return np.allclose(eigenvalues_custom, eigenvalues_numpy, atol=tol)

def main():
    average_iterations = 0
    iteration_counts = []
    for i in range(40):  # Test with 40 different random matrices
        # Set the size of the matrix
        n = np.random.randint(5, 10)  # You can change this value to test with different sizes

        # Generate a random symmetric matrix
        A = random_symmetric_matrix(n)

        # Compare eigenvalues computed by both methods
        if compare_eigenvalues(A):
            eigenvalues_custom, iterations = Rayleigh_shift_deflation_QR(A)
            average_iterations += iterations
            iteration_counts.append(iterations)
        else:
            print("Eigenvalues do not match.")
            eigenvalues_custom, iterations = Rayleigh_shift_deflation_QR(A)
            print("Eigenvalues from custom method:", eigenvalues_custom)
            print("Eigenvalues from NumPy:", np.linalg.eigvals(A))
            break  # Stop testing if a mismatch is found

    print(f"Average iterations: {average_iterations / len(iteration_counts)}")
    print(f"Iteration counts: {iteration_counts}")

if __name__ == "__main__":
    main()
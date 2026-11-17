# Eigenvalue Solver using Shifted QR Algorithm

A Python implementation of the **QR algorithm with Rayleigh quotient shifts and deflation** to compute the eigenvalues of symmetric matrices. To optimize computational efficiency, the input matrix is first reduced to upper Hessenberg form using Householder reflections.

## Features

* **Hessenberg Reduction (`hessenberg`):** Reduces a square matrix to upper Hessenberg form using Householder transformations, significantly speeding up subsequent QR iterations.
* **Rayleigh Quotient Shift & Deflation (`Rayleigh_shift_deflation_QR`):** Accelerates convergence by shifting the matrix using the bottom-right element (\(\mu = H_{m-1, m-1}\)) and deflating the matrix size once an eigenvalue converges within the given tolerance.
* **Automated Validation (`compare_eigenvalues`):** Benchmarks the custom algorithm against NumPy's built-in `np.linalg.eigvals` across 40 randomly generated symmetric matrices to verify accuracy and track iteration counts.

## Prerequisites

* Python 3.x
* NumPy

## Installation & Usage

1. **Clone the repository:**
   ```bash
   git clone https://github.com
   cd ditt-repo-namn
   ```

2. **Install dependencies:**
   ```bash
   pip install numpy
   ```

3. **Run the benchmark script:**
   *(Replace `eigenvalues.py` with your actual filename)*
   ```bash
   python eigenvalues.py
   ```

## Example Output

When executed, the script tests 40 random symmetric matrices (of size 5 × 5 to 9 × 9), verifies that all computed eigenvalues match NumPy's solver within a tolerance of `1e-8`, and outputs convergence statistics:

```text
Average iterations: 12.45
Iteration counts: [11, 14, 9, 15, ...]
```

## Technologies Used

* **Python**
* **NumPy** (Linear algebra & matrix operations)

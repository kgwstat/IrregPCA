# IrregPCA: A fast and flexible approach for PCA with irregularly observed data

## Methodology

The package is based on a risk minimization methodology which minimizes a proper loss function derived from the [Eckhart-Young-Mirsky theorem](https://en.wikipedia.org/wiki/Low-rank_approximation#Basic_low-rank_approximation_problem) for the Hilbert-Schmidt norm. Essentailly, we estimate the functional $$\mathscr{C}[f] = \iint C(u, v) f(u) f(v) \,du\,dv$$ given the data $\mathcal{D} = \{(i, U_{ij}, Y_{ij}): i \in [n], j \in [n_{i}]\}$ where 
$Y_{ij} = X_{i}(U_{ij}) + \varepsilon_{ij}$. 

$$ 
\hat{\mathscr{C}}[f] = \frac{1}{n} \sum_{i= 1}^{n} \left[ 
        \sum_{\substack{p, q = 1\\ p \neq q}}^{n_i} \frac{f(U_{ip}) f(U_{iq}) Y_{ip} Y_{iq}}{n_i (n_i-1)}\right] \\
        - \frac{1}{n(n-1)}\sum_{\substack{i, j = 1\\ i \neq j}}^{n} \left[\sum_{p=1}^{n_i} \frac{f(U_{ip}) Y_{ip}}{n_i} \right]
        \left[\sum_{q=1}^{n_j} \frac{f(U_{jq}) Y_{jq}}{n_j} \right] 
$$ 

## Installation

Install directly from GitHub:

```bash
pip install "git+https://github.com/kgwstat/IrregPCA.git"
```

For local development (editable install):

```bash
pip install -e .
```
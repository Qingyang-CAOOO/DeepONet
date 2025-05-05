<h1 style="text-align: center; font-size: 2em;"> Operator Learning - DeepONet[1] </h1>
<h1 style="text-align: center; font-size: 1.5em;"> Problem 1.A (the antiderivative operator) </h1>

### 1. Operator Learning Formation
- **differential equation**:

    $\frac{ds(x)}{dx} = g(s(x), u(x), x), \quad x \in (0, 1]$

- **initial condition**:

    $$
    s(0) = 0
    $$

- **target mapping**:

    $$
    u(x) \mapsto s(x), \quad \text{for all } x \in [0, 1]
    $$

- **simplification**:
    1. choosing:
        $$
        g(s(x), u(x), x) = u(x)
        $$

    2. the equation became:
        $$
        \frac{ds(x)}{dx} = u(x), \quad s(0) = 0
        $$

        which is the definition of the antiderivative:  

        $$
        s(x) = \int_0^x u(\tau)\, d\tau
        $$

    3. the **operator** $G$ to learn was defined as:

        $$
        G : u(x) \mapsto s(x) = \int_0^x u(\tau)\, d\tau
        $$

- **it's simple and pedagogical**:
    1. **explicit solution**:

        this ODE:$\frac{ds(x)}{dx} = u(x), \quad s(0) = 0$ has a closed-form solution as: $s(x) = \int_0^x u(\tau)\, d\tau$.

    2. **linear operator**:  

        the operator $G : u \mapsto s$ here is linear (they had further nonlinear and stochastic operators).

    3. **no coupling between $s$ and $u$**:  

        In more complex examples (e.g., Problem 1.B), $g$ depends on both $s(x)$ and $u(x)$, introducing feedback and nonlinearity.

    4. **one-dimensional domain**:  

        $x \in [0, 1]$ is just a scalar input-output mapping over a 1D domain — far simpler than PDEs over 2D or 3D spatial domains.

    5. **pedagogical**:
        
        DeepONet learns operator mapping functions to functions, branch net and trunk net separation works, off-line training and on-line inference stages applies...

### 2. DeepONet Architecture

DeepONet is to be trained to approximate the target ground truth:

$$
s(x) = \int_0^x u(\tau)\, d\tau
$$

- **Branch Net**: takes values of $u(x)$ at sensor points $x_1, \ldots, x_m$
- **Trunk Net**: takes an evaluation point $x$
- **Dot Product**: outputs the value $s(x)$

### 3. Implementation Plan
- their implementation: with library DeepXDE
- implementation here: with pytorch
    1. data generator for function pairs
    2. DeepONet model
    3. loss function (MSE)
    4. training
    5. testing

### 4. Implementation Structure
- `models/`: BranchNet, TrunkNet, DeepONet
- `data/`: synthetic dataset for Problem 1.A
- `train.py`: offline training
- `inference.py`: online inference
- `config.py`: hyperparameters
- `utils.py`: seed control

### 5. Training
Run:
```bash
python train.py
```


### References
[1] Lu, L., Jin, P., Karniadakis, G.E. (2021). Learning nonlinear operators via DeepONet based on the universal approximation theorem of operators. *Nature Machine Intelligence*.

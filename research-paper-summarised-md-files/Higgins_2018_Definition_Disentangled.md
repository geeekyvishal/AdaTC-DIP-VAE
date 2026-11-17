# Paper Summary: Towards a Definition of Disentangled Representations (Higgins et al., 2018)

- **Paper ID**: `Higgins_2018_Definition_Disentangled`
- **File**: `1812.02230v1.pdf`
- **Authors**: Irina Higgins, David Amos, David Pfau, Sebastien Racaniere, Loic Matthey, Danilo Rezende, Alexander Lerchner
- **Venue/Year**: arXiv:1812.02230 (2018)

---

## 1. Core Claim
The paper claims that existing intuitive notions of disentangled representations can be formally grounded in modern physics and group representation theory through symmetry transformations. The world $W$ possesses symmetries described by a group $G$ decomposing into a direct product of subgroups $G = G_1 \times \dots \times G_n$. A representation map $f: W \to Z$ into a vector space $Z$ is defined as disentangled with respect to this group decomposition if:
1. $f$ is equivariant ($G$-morphism): $g \cdot f(w) = f(g \cdot w)$ for all $g \in G, w \in W$.
2. The representation space decomposes as a direct sum $Z = Z_1 \oplus \dots \oplus Z_n$ such that each subgroup $G_i$ acts non-trivially only on its corresponding subspace $Z_i$ and acts as the identity on all other subspaces $Z_{j \ne i}$ (Section 1, Section 5, Section 6).

---

## 2. Objective Function (LaTeX)
This paper is a foundational mathematical treatise rather than an algorithmic proposal; it does not introduce a new variational loss function. The core mathematical formulation is the **Equivariant Disentangled Group Action** (Section 5.1, Equation 1; Section 5.2, Equation 2):

$$(g_1, g_2) \cdot (v_1, v_2) = (g_1 \cdot_1 v_1, g_2 \cdot_2 v_2)$$

$$g \cdot f(w) = f(g \cdot w) \quad \forall g \in G, w \in W$$

In terms of linear group representations $\rho: G \to GL(V)$ over representation space $V = V_1 \oplus \dots \oplus V_n$ (Section 6.1–6.2):
$$\rho(g_1, \dots, g_n) = \rho_1(g_1) \oplus \dots \oplus \rho_n(g_n)$$
where each subrepresentation $\rho_i$ acts exclusively on subspace $V_i$.

Verbatim quote from Section 5.1, Page 11:
> "Suppose that we have a group action $\cdot: G \times X \to X$, and that the group $G$ decomposes as a direct product $G = G_1 \times G_2$. We are going to refer to the action of the full group as $\cdot$, and the actions of each subgroup as $\cdot_i$. Then we propose the following definition: the action is disentangled (with respect to the decomposition of $G$) if there is a decomposition $X = X_1 \times X_2$, and actions $\cdot_i : G_i \times X_i \to X_i, i \in \{1,2\}$ such that
> $(g_1,g_2) \cdot (v_1,v_2) = (g_1 \cdot_1 v_1, g_2 \cdot_2 v_2) \quad (1)$
> In particular this says that an element of $G_1$ acts on $X_1$ but leaves $X_2$ fixed, and vice versa."

Verbatim quote from Section 5.2, Page 12:
> "In other words, we want the action on $Z$ to correspond to the action on $W$. This can be achieved if the following condition is satisfied:
> $g \cdot f(w) = f(g \cdot w) \quad \forall g \in G, w \in W \quad (2)$
> This states that the action should commute with $f$. This is the definition of a $G$-morphism or an equivariant map."

---

## 3. Which KL Term It Modifies
- **Modifications**: N/A (foundational theory paper).
- **Theoretical Relevance**: Formalizes why standard 1D scalar factorization (Total Correlation minimization) is mathematically limited: independent factors of variation correspond to commuting subgroup factors ($G_1 \times G_2$). Non-commutative transformations (such as 3D spatial rotations $SO(3)$) cannot be factored into 1D independent coordinates and require multi-dimensional irreducible subspaces.

---

## 4. Total Correlation (TC) Estimator Used
- **TC Estimator**: N/A.

---

## 5. Datasets, Metrics, and Results
- **Illustrative Systems**:
  - Discrete Grid World: Agent in a grid undergoing 2D cyclic translations ($G_x \times G_y$) and color/hue shifts ($G_c$) (Section 5, Figure 1A, Figure 2).
  - 3D Rotations $SO(3)$: Demonstrating that non-commuting rotations along different spatial axes cannot be decomposed into a direct product $G_1 \times G_2$, but the full rotation group $SO(3)$ can be disentangled from independent attributes like color or scale (Section 5, Figure 1B).
- **Evaluated Models**: Visualizes representation subspaces learned by CCI-VAE (Burgess et al., 2018), verifying that learned trajectories approximately preserve subgroup commutativity and compositionality (Figure 2).

---

## 6. Limitations
1. **No Algorithmic Training Objective**: Provides a formal definition and diagnostic framework, but does not provide a practical unsupervised training objective for continuous neural network optimization.
2. **Assumption of Group Structure**: Assumes data transformations form clean mathematical groups (invertible, associative, containing identity), which may not hold for non-invertible physical processes (e.g. occlusion, projections, non-reversible deformations).

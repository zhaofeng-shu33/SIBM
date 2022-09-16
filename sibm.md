## How to draw X in two communities

$X$ is draw from uniformly whose $\binom{n/2}{n}$ position takes value $+1$;

Each component of $X$ is draw from Bernoulli distribution with $p=0.5$.

Abbe said they were equivalent when $n$ is large.

## A simple illustration of Abbe's conclusion (2015)

$$
p=\frac{a\log n}{n}, q=\frac{b\log n}{n}
$$

Fully recovering requires $\sqrt{a} - \sqrt{b} > \sqrt{2}$

Problems with critical value?

Min Ye:
$$
P(B_i - A_i \geq 0) \leq n^{-\frac{(\sqrt{a} - \sqrt{b})^2}{2}}
$$
MNS16 (Theorem 2.5):

​      $X \sim B(\frac{n}{2}, p_n), Y \sim B(\frac{n}{2}, q_n), \Pr(Y\geq X) = o(n^{-1}) \iff $ fully recovery

## Extension of Ising model to three communities

### critical value of $\beta$

$\sigma \in \{1,\omega,\omega^2 \}^n$, root of $x^3-1$. cyclic group: $\sigma^3=1$

![](./case3.png)

Indicator function
$$
I(\sigma_i, \sigma_j) = \begin{cases} 2 & \sigma_i = \sigma_j \\
-1 & \sigma_i \neq \sigma_j\end{cases}
$$
Notice that $I(\sigma_i, \sigma_j) = 2 \cdot 1_{\sigma_i = \sigma_j} -1 $. Choosing either indicator function is equivalent.



Suppose $x_i, \dots, x_n$ are draw uniformly from $\{1,\omega,\omega^2\}^n$ (That is, there are $\frac{n}{3}$ items taking value $0$, $\frac{n}{3}$ items taking value 1...)

Thus guarantee $\sum_{i=1}^n I(x_i, \sigma) = 0$ for any $\sigma$

Ising model
$$
P_{\sigma | G}(\sigma = \bar{\sigma}) = \frac{1}{Z_G(\alpha, \beta)}\exp\left(\beta \sum_{\{i,j\}\in E(G)}I(\bar{\sigma}_i, \bar{\sigma}_j) - \frac{\alpha \log(n)}{n} \sum_{\{i,j\} \not\in E(G)} I(\bar{\sigma}_i,\bar{\sigma}_j)\right)
$$




For SBM with three communities and $p=\frac{a\log n}{n}, q=\frac{b\log n}{n}$

  $\sqrt{a} - \sqrt{b} > \sqrt{3}$ is sufficient to recover the community structure (Abbe).

Define
$$
\begin{align}
A_i &= |\{j \in [n]\backslash\{i\}: \{i,j\} \in E(G), X_j = X_i\}| \\
B_i &= |\{j \in [n]\backslash\{i\}: \{i,j\} \in E(G), X_j = \omega \cdot X_i\}| \\
C_i &= |\{j \in [n]\backslash\{i\}: \{i,j\} \in E(G), X_j = \omega^2 \cdot X_i\}|
\end{align}
$$


Computation of critical value $\beta^*$:
$$
\begin{align}
\frac{P_{\sigma |G}(\sigma_i \neq X_i)}{P_{\sigma | G}(\sigma = X)} &=
\frac{P_{\sigma |G}(\sigma_i = \omega \cdot X_i)}{P_{\sigma | G}(\sigma = X)}+\frac{P_{\sigma |G}(\sigma_i = \omega^2 \cdot X_i)}{P_{\sigma | G}(\sigma = X)}\\
&= \exp\left(3(\beta + \frac{\alpha \log n}{n})(B_i-A_i)-\frac{4\alpha \log n}{n}\right)\\
&+ \exp\left(3(\beta + \frac{\alpha \log n}{n})(C_i-A_i)-\frac{4\alpha \log n}{n}\right)
\end{align}
$$
Due to symmetry, when evaluation the expectation, only

need to examine $\mathbb{E}_G[\exp(3\beta (B_i - A_i))]$ when $A_i \sim B(\frac{n}{3}-1, \frac{a\log n}{n}), B_i \sim B(\frac{n}{3}, \frac{b\log n}{n})$.
$$
g(\beta) = \frac{1}{3}(ae^{-3\beta}+be^{3\beta}) - \frac{a+b}{3} +1
$$
$g'(\beta)$ has unique root $\frac{1}{6}\log \frac{a}{b}$ and $g(\beta)$ is convex. and $g(\frac{1}{6} \log \frac{a}{b}) = \frac{1}{3}(2\sqrt{ab}-a-b) + 1$.

Let $g(\frac{1}{6} \log \frac{a}{b}) < 0$ we get the threshold: $\sqrt{a} - \sqrt{b} > \sqrt{3}$, which is same with Abbe's result.

The smaller root of $g(\beta)=0$ is
$$
\beta^* = \frac{1}{3}\log \frac{a+b-3 - \sqrt{(a+b-3)^2-4ab}}{2b}
$$

### Generalization of $\alpha = b \beta$

Empirical deduction: This result can be got by evaluating $\mathbb{E}_G \log(\Pr(\sigma = X)) = \mathbb{E}_G \log(\Pr(\sigma = 1)) $,

To be more precisely:

Consider the expectation of edge numbers: 
$$
\mathbb{E}_{G}[|E|] = \frac{a \log n}{n} \frac{n}{2}(\frac{n}{2} - 1) + \frac{b\log n}{n}\left(\frac{n}{2}\right)^2 \approx \frac{a+b}{4} n \log n
$$
Then 
$$
\begin{align}
\mathbb{E}_G \log(\Pr(\sigma = 1)) &= \beta \mathbb{E}_{G}[|E|] - \frac{\alpha \log n}{n}(\frac{n^2-n}{2}-\mathbb{E}_{G}[|E|])-\log Z \approx (\frac{(a+b)\beta}{4} - \frac{\alpha}{2})n\log n \\
\mathbb{E}_G \log(\Pr(\sigma = X)) &= \frac{\beta}{2}\sum_{i=1}^n(\mathbb{E}[A_i - B_i]-\frac{\alpha \log n}{n}\mathbb{E}[(\frac{n}{2}-1-A_i)-(\frac{n}{2}-B_i)])\approx \frac{(a-b)\beta}{4}n\log n
\end{align}
$$
Therefore we can get $\alpha = b \beta$ as the critical condition.

Using the same method as above, we can get that $\alpha = b \beta$ is also critical value for $k=3$.

With $\mathbb{E}_G[|E|] = \frac{n\log n}{3}(\frac{a}{2} + b)$, $\mathbb{E}_G \log(\Pr(\sigma = 1))\approx (\frac{a+2b}{3}\beta - \alpha) n \log n$ and $\mathbb{E}_G \log(\Pr(\sigma = X)) \approx \frac{\beta(a-b)}{3} n \log n $

To be more rigorous, we would like to estimate

 $\sum_{i,j} M_{ij} I(\sigma_i, \sigma_j)$  where $M_{ij}$ is defined as
$$
M_{ij} =
\begin{cases}
(a\beta - \alpha + a\alpha \frac{\log n}{n}) \frac{\log n}{n} &\textrm{ if } X_i = X_j \\
(b\beta - \alpha + b\alpha \frac{\log n}{n}) \frac{\log n}{n} &\textrm{ if } X_i \neq X_j 
\end{cases}
$$
Let
$$
\Xi = \begin{bmatrix}
u_1, u_2, u_3 \\
v_1, v_2, v_3 \\
w_1, w_2, w_3
\end{bmatrix}, Q = \begin{bmatrix}
Q_{11}, Q_{12}, Q_{13} \\
Q_{21}, Q_{22}, Q_{23} \\
Q_{31}, Q_{32}, Q_{33}
\end{bmatrix}
$$
which has 9 non-negative variables. Besides, each row of $\Xi$ sums to $\frac{n}{3}$.

The meaning of each variable:

$u_1 = |\{i\in [n]: X_i =1, \sigma_i = 1\}|$, $u_2 = |\{i\in [n]: X_i =1, \sigma_i = \omega\}|$, $u_3 = |\{i\in [n]: X_i =1, \sigma_i = \omega^2\}|$,$v_1 = |\{i\in [n]: X_i =\omega, \sigma_i = 1\}|$.

Suppose the subscript starts from 1: $\Xi_{ij} = |\{k \in [n]: X_k = \omega^{i-1}, \sigma_k = \omega^{j-1}\}|$.

Also
$$
Q_{ij} = (\Xi_{i1} - \Xi_{i2})(\Xi_{j1} - \Xi_{j2}) + (\Xi_{i1} - \Xi_{i3})(\Xi_{j1} - \Xi_{j3}) + (\Xi_{i2} - \Xi_{i3})(\Xi_{j2} - \Xi_{j3})
$$


$A =(a\beta - \alpha) \frac{\log n }{n}, B=(b\beta - \alpha) \frac{\log n}{n}$

We can write (the version dose not dividing 2)
$$
\sum_{i,j=1}^n M_{ij} I(\sigma_i, \sigma_j) = (Q_{11} + Q_{22} + Q_{33})(A-B) + (\sum_{i,j=1}^3 Q_{ij})B + \textrm{ high order term}
$$
Using the expression of $Q_{ij}$ we can show that the coefficient of $B$ is
$$
\begin{align}
\sum_{i,j=1}^3 Q_{ij} = &((u_1+v_1+w_1)-(u_2+v_2+w_2))^2 \\
+ &((u_1+v_1+w_1)-(u_3+v_3+w_3))^2 + ((u_2+v_2+w_2)-(u_3+v_3+w_3))^2
\end{align}
$$
which is non-negative.

Therefore, when $\alpha > b \beta (B<0)$, we should let $\sum_{i,j=1}^3 Q_{ij}=0$ to maximize $\sum_{i,j} M_{ij} I(\sigma_i, \sigma_j)$.

Actually we can maximize the coefficient of $A$ and $B$ respectively and these two maximal values can co-exist.

There are 6 kinds of possibilities for such $\Xi \in \{e, s, r, r^2, sr, sr^2\}$. $e=\frac{n}{3}I_3$ is the identity of the group.
$$
r = \frac{n}{3}\begin{bmatrix}
0 & 1 & 0 \\
0 & 0 & 1 \\
1 & 0 & 0
\end{bmatrix}, s = \frac{n}{3} \begin{bmatrix}
0 & 1 & 0 \\
1 & 0 & 0 \\
0 & 0 & 1
\end{bmatrix}
$$


This group is isomorphic to $S_3$.

This is equivalent to say when $\sigma  = f(X)$, $\Pr(X=\sigma)$ is maximized under the condition of $\alpha > b \beta$.

The function $f$ is applied to $X$ in pairwise. For each element, it is selected from the symmetric group $S_3$,

for example
$$
f=\binom{1, \omega, \omega^2}{\omega, 1, \omega^2}
$$
means $f(1) = \omega, f(\omega)=1, f(\omega^2) = \omega^2$.

When $\alpha < b \beta$, similarly we can show that when $\sigma$ all equals to $1, \omega$ or $\omega^2$, the maximum is taken. We represent it as $f_1$.

## How to show the concentration of $X_G$

$$
X_G = \beta \sum_{\{i,j\} \in E(G)} I(\sigma_i, \sigma_j) - \frac{\alpha \log n}{n}\sum_{\{i,j\} \not\in E(G)} I(\sigma_i, \sigma_j)
$$

For $k=3$, we have $I(\sigma_i, \sigma_j) = 2 \mathrm{Re}(\sigma_i \bar{\sigma}_j)$. $\bar{\sigma}$ is the complex conjugate of $\sigma$. 

This result cannot be extended to $k>3$ using complex plane.

A simple illustration for $k=4$ (the number is $\sigma_i \bar{\sigma}_j$ when $\sigma_i \neq \sigma_j$:

![](./impossible_4.svg)

Needs at least a tetrahedron in 3D.

But for $k=3$, we can still use the matrix conclusion for complex inner space:
$$
|\mathrm{Re}(\bar{\sigma}^T(A-\mathbb{E}[A|X])\sigma)| \leq cn\sqrt{\log n}
$$
$\sigma$ is a column complex vector here and the above result holds almost surely.

Then all we need to do is to make sure that when $\sigma$ is a little bit far from the six **corner** maximum value,

Then the probability is much smaller than the maximal one.

Define a small neighborhood of for a corner value $\Xi^*$ (each row of $\Xi$ has only one non-zero term).
$$
N(\Xi^*) := \{\Xi |\, |\Xi_{i,j} - \Xi_{i,j}| \leq \frac{n}{\log^{1/3} n}, 1\leq i,j\leq 3\}
$$
We use the $L_{\infty}$ norm for the matrix.

There are 27 different corner values.

For those $\Xi$ falling into the corner neighborhood, we can show the coefficient of $(A-B)$ is very near to the maximal value $\frac{2}{3}n^2$. However, when $\Xi$ falls out of these 6 maximum corner neighborhoods, the coefficient of $B$ is larger than $(\frac{n}{3} - 3\frac{n}{\log^{1/3} n})^2$. When $n$ is sufficient large, this gap is linear with $n^2$, thus contributing to
$$
\frac{P_{\sigma|G}(\sigma = \sigma')}{P_{\sigma|G}(\sigma=X)} < 3^{-n}e^{-n}
$$
Another case happens when the coefficient of $A-B$ has a big gap with the maximal value $\frac{2}{3}n^2$. When $\Sigma$ is not a corner value. We can find a row of $\Xi$ such that $\frac{2}{3}n^2 - Q_{ii} \geq \frac{4n^2}{3 \log^{1/3} n}$. Using this gap the above inequality also holds.



We get the conclusion that when $\alpha > b \beta$, $\mathrm{dist}(\sigma^{i}, f(X)) < 3 n / \log^{1/3} (n) $ almost surely.

## concentration of general k(Professor Ye)

In this section we assume
$$
I(x ,y) = \begin{cases}
k-1 & x = y\\
-1 & x \neq y
\end{cases}
$$
Then we show that $|\sum_{i,j} I(\sigma_i, \sigma_j) (A-\mathbb{E}[A|X])| \leq cn\sqrt{\log n}$ almost surely.

Since there are k communities $\{\omega^0, \omega^1, \dots, \omega^{k-1}\}$, we consider 2 communities $\omega^{r}, \omega^s$ at each step. Let 
$$
\gamma^{r,s}_i = \begin{cases}
1 & \sigma_i = \omega^r \\
-1 & \sigma_i = \omega^s \\
0 & \textrm{ otherwise}
\end{cases}
$$
Then it is not difficult to show that
$$
\sum_{i,j} I(\sigma_i, \sigma_j) (A-\mathbb{E}[A|X]) = \sum_{r,s} (\gamma^{r,s})^T (A-\mathbb{E}[A|X])\gamma^{r,s}
$$
We decompose the flatten summation to that of quadratic product.

Then we can use the concentration result for the matrix norm $||A-\mathbb{E}[A|X]||$:

With probability $1-n^{-r}$:
$$
|\sum_{r,s} (\gamma^{r,s})^T (A-\mathbb{E}[A|X])\gamma^{r,s}| \leq \sum_{r,s} ||A-\mathbb{E}[A|X]|| \cdot||\gamma^{r,s}||^2 \leq \binom{k}{2} cn\sqrt{\log n} 
$$


## Proof for $\beta \in (\beta^*, \frac{1}{6}\log \frac{a}{b})$

When $\beta$ in this range, we will show that one sample is enough to recover the original label $X$.

For each $I$, we introduce a vector $v$ with $\mathrm{dim}(v) = |I|=k$. Each element of $v$ takes from $\{\omega, \omega^2\}$.

$X^{(\sim I, v)}$ is defined as flipping $i \in I$ as : $X_i \to v_{\mathrm{index}(i)}\cdot X_i$.

For example, if $I = {1,3}, n=3, v=(\omega^2, \omega)$ then $X^{(\sim I,v)} = (\omega^2 \cdot X_1, X_2, \omega \cdot X_3)$.

Define $P_{\sigma|G}(\sigma = X^{(\sim I)})=\sum_{v}P_{\sigma|G}(\sigma = X^{(\sim I,v)})$

Using this extra structure, we can get
$$
\frac{P_{\sigma|G}(\sigma = X^{(\sim I)})}{P_{\sigma |G}(\sigma =X)} \leq 2^k \exp(3(\beta + \frac{\alpha \log n}{n})(B_{I,v} - A_I)))
$$
where
$$
\begin{align}
A_I &= |\{i \in I,j \in [n]\backslash I: \{i,j\} \in E(G), X_j = X_i\}| \\
B_{I,v} &= |\{i \in I, j \in [n]\backslash I: \{i,j\} \in E(G), X_j = v_{\mathrm{index}(i)} \cdot X_i\}| \\
\end{align}
$$

$$
\mathrm{E}[\exp(3(\beta + \frac{\alpha \log n}{n})(B_{I,v} - A_I)))] \sim n^{k(g(\beta)-1)}
$$

The difference lies at the term $2^k$, which does not influence the proof using the formula of geometric series.

## Proof for $\beta > \beta^*$

Define
$$
\tilde{g}(\beta) := \begin{cases}
g(\beta) & \beta < \frac{1}{6}\log \frac{a}{b} \\
g(\frac{1}{6}\log\frac{a}{b}) & \beta \geq \frac{1}{6}\log\frac{a}{b}
\end{cases}
$$
$\tilde{g}(\beta) < 0 $ for $\beta > \beta^*$ when $\sqrt{a} - \sqrt{b} > \sqrt{3}$.

We first show that for almost $G$
$$
\sum_{i = 1}^n \frac{P_{\sigma | G}(\mathrm{dist}(\sigma, X^{(\sim I,v)})=1)}{P_{\sigma |G}(\sigma = X)} < 2n^{k\tilde{g}(\beta)/2}
$$
Notice the coefficient $2$ is due to $v\in \{\omega, \omega^2\}$. We need only to consider the half part: $P_{\sigma | G}(X_i \neq \omega \cdot \sigma_i)$ in the numerator.

In such case:
$$
\mathbb{E}[\tilde{D}(G)] \leq n^{1-\frac{(\sqrt{a}-\sqrt{b})^2}{3} + o(1)}
$$
Also $P(\tilde{D}(G) = 0) = 1 - o(1)$

Proposition 5*

For $t \in [\frac{1}{3}(b-a), 0]$, $B_i \sim B(\frac{n}{3}, \frac{b\log n}{n}), A_i \sim B(\frac{n}{3}, \frac{a \log n}{n})$.

Then
$$
P(B_i - A_i \geq t \log n) \leq \exp\left(\frac{\log n}{3}(\sqrt{9t^2 + 4ab} - 3t\log\frac{\sqrt{9t^2+4ab}+3t}{2b} - a - b + O(\frac{\log n}{n}))\right)
$$

$$
f_{\beta}(t):=\frac{1}{3}\sqrt{9t^2 + 4ab} - t\log\frac{\sqrt{9t^2+4ab}+3t}{2b} - \frac{a + b}{3} + 1 + 3 \beta t
$$

We need to show $f_{\beta}(t) \leq \tilde{g}(\beta)$ for $ t < 0$.

## Multiple sample case

Algorithm:

Step 1: Alignment, assuming $\sigma^{(1)}$is ground truth. Using the six permutation functions and select the one with maximum alignment(after permutation nearest to $\sigma^{(1)})$ . It can be shown that $d(f(\sigma^{(j)}), \sigma^{(1)}) \leq \frac{n}{3}$.

## 

Score.

Step 2: Majority vote at each coordinate

$\hat{X}_i = \max\{|\{j | \sigma^{(j)}_i = 1,1\leq j \leq m\}|,|\{j | \sigma^{(j)}_i = \omega,1\leq j \leq m\}|,|\{j | \sigma^{(j)}_i = \omega^2,1\leq j \leq m\}|\}$



## weak discovery in SIBM

Let $(X, G, \{\sigma^{(1)}, \dots, \sigma^{(m)}\}) \sim SIBM(n, \frac{a}{n}, \frac{b}{n}, \beta, m)$

If there exists an algorithm that takes $ \{\sigma^{(1)}, \dots, \sigma^{(m)}\}$ as inputs and outputs $\hat{X} =\hat{X}( \{\sigma^{(1)}, \dots, \sigma^{(m)}\})$ such

that

$ P(A(\hat{X}, X) \geq 1/2 + \epsilon)  \to 1$ as $n \to \infty$

For SBM ($\frac{a}{n}, \frac{b}{n}$)with two communities, the problem is solvable if $(a-b)^2 > 2(a+b)$.

Not solvable if $(a-b)^2 \leq 2(a+b)$ (Mossel 2015)

Efficient algorithm for weak discovery scenario: k-cycle

## Voter model

https://en.wikipedia.org/wiki/Sznajd_model

The idea of this model is mentioned in SIBM paper.

## Metropolis

Using Metropolis algorithm to generate the SIBM sample: https://en.wikipedia.org/wiki/Ising_model#Metropolis_algorithm



Hamiltonian of SIBM
$$
H(\bar{\sigma}) = \frac{\alpha}{\beta} \frac{\log n}{n} \sum_{\{i,j\} \not\in E(G)}\bar{\sigma}_i\bar{\sigma}_j - \sum_{\{i,j\} \in E(G)} \bar{\sigma}_i \bar{\sigma}_j
$$
If there is one coordinate of $\bar{\sigma}$ flips: $\bar{\sigma}_r \to -\bar{\sigma}_r$. Then the change of energy is:
$$
H(\bar{\sigma}^{\sim\{r\}}) - H(\bar{\sigma}) = 2 \sum_{i \in N_r(G)} \bar{\sigma}_r \bar{\sigma}_i - 2\frac{\alpha \log n}{\beta n} \sum_{i \in V\backslash N_r(G)} \bar{\sigma}_r \bar{\sigma}_i
$$
We choose the index $r$ randomly from $V$ (in implementation we iterate over all vertices one by one) and calculate the energy difference, if $H(\bar{\sigma}^{\sim\{r\}}) < H(\bar{\sigma})$ we accept the new configuration. Otherwise, we accept it with probability $\exp^{-\beta (H(\bar{\sigma}^{\sim\{r\}}) - H(\bar{\sigma}))}$. We repeat the

iteration for a given number of times.

Simulation problem:



## A naive parameter estimation method

Given a graph $G$, we can count its edges as $|E|$.

The first equation is 
$$
\frac{n}{2}(\frac{n}{2} - 1) p + (\frac{n}{2})^2 q= |E|
$$
This equation gives the value of $p+q$ approximately.

Then we choose two nodes $A$ and $B$:

$H_1$: the labels of $A$ and $B$ are the same;

$H_2$: the labels of $A$ and $B$ are opposite;

To distinguish $H_1$ from $H_2$, we use two statistics $T_1$ and $T_2$.

$T_1$: the number of nodes which have edges to $A$ and $B$ or do not have edges to $A$ and $B$.

$T_2$: the number of nodes which have only one edges to $A$ or $B$.

Under the hypothesis of $H_1$: 

$E[T_1] \approx \frac{n}{2} (p^2 + q^2 + (1-p)^2 + (1-q)^2)$

$E[T_2] \approx n(p(1-p) + q(1-q))$

Under the hypothesis of $H_2$:

$E[T_1] \approx n (pq + (1-p)(1-q))$

$E[T_2] \approx n(p(1-q) + (1-p)q)$

Notice that $E[T_1] + E[T_2] \approx n$ under both cases.

The problem is that $E[T_1] \approx n$ under both cases and it is hard to distinguish $H_1$ from $H_2$.

Actually, we can count the triangles in the graph, and get an equation of 3rd order:
$$
\frac{n/2(n/2-1)(n/2-2)}{3}p^3+ (n/2)^2(n/2-1)pq^2 = |T| 
$$
After some simplification, we can get the equation array of $a$ and $b$.

Question: in ER graph, as $n\to \infty$ how to show that $6\frac{|T|}{(log n)^3} \to a$  where $p=\frac{a \log n}{n}$.

The sum of $\frac{n^3}{6}$ of Bernoulli random variables with parameter $p^3$, but they are not independent.



Other parameter estimation method

It is mentioned in 4.1 of [1] that suppose $\sigma \in \{\pm\}^n$ is the true labels and $A$ is adjacency matrix, then
$$
E[A|\sigma] = \frac{p + q}{2} 11^T + \frac{p-q}{2}\sigma \sigma^T
$$
We can compute the second largest eigenvalues of $E[A|\sigma]$.



## Reference

[1] Mossel, Elchanan, Joe Neeman, and Allan Sly. "Consistency thresholds for the planted bisection model." *Proceedings of the forty-seventh annual ACM symposium on Theory of computing*. 2015.

[2] Abbe, Emmanuel, Afonso S. Bandeira, and Georgina Hall. "Exact recovery in the stochastic block model." *IEEE Transactions on Information Theory* 62.1 (2015): 471-487.

[3] Mossel, Elchanan, Joe Neeman, and Allan Sly. "Reconstruction and estimation in the planted partition model." *Probability Theory and Related Fields* 162.3-4 (2015): 431-461.
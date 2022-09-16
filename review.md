Literature Review on Recent Developments on the topic of community discovery(theoretical literature)

## Ising Block Model (2019)

* No stochastic assumption about underlining community.

* Consider two communities

* Get the critical region involving $\alpha, \beta$.

Generating Model: Consider a partition of $V=\{1,\dots, p\}$ into $S$ and $\bar{S}$ with $|S|=\frac{p}{2}$, a determined graph $G$ is the combination of two complete graphs of $S$  and $\bar{S}$. Then we can generate a probability distribution
$$
P_{\sigma | G}(\sigma = \bar{\sigma}) = \frac{1}{Z_G(\alpha, \beta)}\exp\left(\frac{\beta}{2p} \sum_{\{i,j\}\in E(G)}\bar{\sigma}_i \bar{\sigma}_j + \frac{\alpha}{2p} \sum_{\{i,j\} \not\in E(G)} \bar{\sigma}_i \bar{\sigma}_j\right), \bar{\sigma}_i \in \{\pm 1 \}
$$
$\beta > \alpha$ can guarantee that inner community attraction is larger than intra-community one. ($\alpha$ can be positive or negative)

In this paper, the critical region of $\alpha$ and $\beta$ is got. Under these two regions, the sample complexity of

independent samples $n$ which is necessary to recover $S$ with probability 1 has been deducted.

## Hypergraph stochastic block model (2020)

Consider the recovery problem of k clustering for d-uniform hypergraph. That is, every edge connects $d$ nodes. This paper focus on the regime when $p,q$ are fixed while $n,k$ tends to infinity (dense community).

## Weighted Stochastic Block Model (2020)

Each edge is associated with a weight from a given density function $p(\cdot)$ (if the edge is within the community) or $q(\cdot)$ (if the edge joins two different communities).

The result is about general weighted model and some strange assumptions have to be used to make proof possible. For example, this theory requires each cluster should have at least $\frac{n}{\beta k}$ nodes. The extra parameter $\beta$ is introduced.

The misclustering error used is the percentage of wrong results.

## Hierarchical Model proposed by Professor Huang

Consider the case with 2 attributes, there are four communities $S_1, S_2, S_3, S_4$, each with equal size $|S_i|=\frac{n}{4}$.

For $(i,j) \in S_i$, the probability of existence of an edge is $p$;

For $i \in S_1, j \in S_2$ or $i\in S_2, j\in S_3$ or $i\in S_3, j\in S_4$ or $i\in S_4, j\in S_1$ , the probability of existence of an edge is $q$;

For $i\in S_1, j\in S_3$ or $i\in S_2, j \in S_4$, the probability of existence of an edge is $r$. We require that $p>q>r$

Under which condition of $p,q,r$ can we find an efficient algorithm to recover the community as $n\to \infty$?

When $q=r$, the model reduces to SBM with $k=4$.

![](./S4.svg)

Practical Implications: We can think each node is associated with two binary attributes. When the two attributes are the same, they belong to one cluster $S_i$. When only one attribute differs, the relationship is

captured by $q$; When both the two attributes differ, the quantity is described by $r$.



[1] Berthet Q, Rigollet P, Srivastava P. Exact recovery in the Ising blockmodel. Ann Statist. 2019;47(4):1805-1834.

[2] Cole S, Zhu Y. Exact recovery in the hypergraph stochastic block model: A spectral algorithm. Linear Algebra and its Applications. 2020;593:45-73.

[3] Xu M, Jog V, Loh PL. OPTIMAL RATES FOR COMMUNITY ESTIMATION IN THE WEIGHTED STOCHASTIC BLOCK MODEL. Annals of Statistics. 2020;48(1):183-204.

# Literature Review on Semi-supervised SBM (Algorithm)

## Signed networks Model (2020)

* partially labeled node
* the edge takes values from $\{\pm 1\}$
* Variational Bayesian




## Empirical Study of Phase transitions (2014)

- SBM with a fraction $\alpha$ nodes whose labels are known
- Study the detection versus non-detection regime using a belief propagation algorithm

## Belief propagation is optimal in some region when side information is provided (2016, Mossel)

Let $\sigma$ be true label, each component of $\sigma$ is sent into a BSC channel with probability $\alpha \in [0,1/2)$.

Then we can get a noisy label $\tilde{\sigma}$. Using the side information $\tilde{\sigma}$ and consider a SBM$(\frac{a}{n}, \frac{b}{n})$.

[1] Liu, X., Song, W., Musial, K., Zhao, X., Zuo, W., Yang, B., 2020. Semi-supervised stochastic blockmodel for structure analysis of signed networks. Knowledge-Based Systems.. doi:10.1016/j.knosys.2020.105714

[2] Zhang, Pan, Cristopher Moore, and Lenka Zdeborová. "Phase transitions in semisupervised clustering of sparse networks." *Physical Review E* 90.5 (2014): 052802.

[3] Mossel, Elchanan, and Jiaming Xu. "Local algorithms for block models with side information." *Proceedings of the 2016 ACM Conference on Innovations in Theoretical Computer Science*. 2016.



# Information Theoretic Approach in Community Detection

## pairwise measurement

The sample complexity (the number of edges required) to almost exactly recover the labels up to a constant

is of order $\frac{n\log n}{\mathrm{Hel}^{\mathrm{min}}_{1/2}}$.

Also, the information measure is difficult to estimate from one sampling of data.

[1] Chen, Yuxin, Changho Suh, and Andrea J. Goldsmith. "Information recovery from pairwise measurements." *IEEE Transactions on Information Theory* 62.10 (2016): 5881-5905.

## New model based on pairwise measurement

Consider ER graph with $p_{\textrm{obj}} \geq \log n /n$. Every node of the graph has a label $x_i$ coming from a cyclic group with order $M$. Suppose for every edge of the graph, there is an observation value $y$ coming from $\{0, 1, \dots, M\}$ and is prescribed by $p(y_{ij} | x_i - x_j = l) = P_{l}(y_{ij})$.  Also the discrete 

conditional distribution is only dependent on $l$.  Also for each node there are multiple independent samples (the number of sample is T) with probability $p(y_i | x_i = l)= P_l(y_i)$. The transfer probability is the same with that of edges. Notice node sample is different with edge sample. For each edge we only have one observation but for each node we have multiple observations. Then we investigate the error probability of ML algorithm $\psi$ to recover $X$. Also notice that the error probability is chosen as the maximum error:
$$
P_e(\psi) := P\{\textrm{dist}(\psi(y), x) \neq 0 | x \}
$$
The chosen of multiple observations for each node is to balance the contribution of nodes and that of edges. Compared to the original pairwise measurements model, we added noisy node observations, making the new model semi-supervised learning.

Some ideas to transform node observation to edge observation: Add new nodes with label zero. And add one edge from the new node to each old node.

Problems: the transformation is deterministic and the expanded graph does not satisfy the condition any more.

We take an approach to insert additional observations into the proof of Theorem 1 in the original paper of Chen, Yu Xin. 

$N_w = T n_0 + \frac{1}{2}(n^2 - \sum_{i=0}^{M-1} n_i^2 )$

In Chen's original paper,  there is an assumption of $n_0 \geq n_i$ for $i=1, \dots, M-1$. (Equation 91).

This assumption is possible due to the property of translational invariant. We can consider only

proposals with $n_0 \geq n_i$ and multiply the error probability by $M$.

### New model based on SBM

Consider $n$ nodes $Y_1, \dots, Y_n$ with binary labels. If  $Y_i  = Y_j$ there are probability $p$ that there is an edge between the two nodes; If $Y_i \neq Y_j$, there are probability $q$ that there is an edge between them. Also $p>q$.

For each node $Y_i$, we can generate $m$ i.i.d. observations $X_1^{(i)}, \dots, X_m^{(i)}$ from $P_0$ (if $Y_i = 0$) or $P_1$ (if $Y_i=1$).

Suppose given one sample of the graph and $X_j^{(i)}$ for $i=1, \dots, n$ and $j=1, \dots, m$,

What is the Chernoff Bound for ML method?

For SBM only, it is proved that the error probability $\leq c^{-\frac{1}{4}\epsilon}$.

If there is no graph structure, then we do a Hypothesis testing independently for each node, and the judging scheme is to compare $P(X^{(i)}_1, \dots, X^{(i)}_m | Y_i=0)=\prod_{j=1}^m p_0(x^{(i)}_j)$ and $P(X^{(i)}_1, \dots, X^{(i)}_n | Y_i=1) = \prod_{j=1}^m p_1(x_j^{(i)})$ and the error probability decreases as $\exp^{-n D}$ where $D$ is called the Chernoff component and can be computed from the two joint distribution.

Equivalence of ML in SBIM to bisection partition.

Let $z_{ij} \in \{0, 1\}$ to represent whether there is an edge between two nodes in a graph, then
$$
 
$$
Let $A$ to represent the number of edges between two parts $y_i=1$ and $y_i=0$.

Then 
$$
p(z|y)=p^{|E|-A}(1-p)^{\frac{n}{2}(\frac{n}{2}-1)-|E|+A} q^A (1-q)^{\frac{n^2}{4}-A}
$$
Suppose $p>q$, to maximize $p(z|y)$ is equivalent to minimize $A$. That is, to find a bisection which minimizes the number of edges across the cut.

Consider also $p(x|y)$ as:
$$
p(x|y) = \prod_{i=1}^n \prod_{j=1}^m p_0(x_j^{(i)})^{y_i} p_1(x_j^{(i)})^{1-y_i}
$$
We maximize $\log p(x,z | y) = \log p(z | y) + \log p(x|y) = C + \sum_{i=1}^n B_i y_i + (\log q - \log p - \log (1-q) + \log (1-p)) A$



The coefficient of $A$ is negative, $C$ is a constant, not involving $y_i$ and $B_i  = \sum_{j=1}^m \log \frac{p_0(x_j^{(i)})}{p_1(x_j^{(i)})}$.

Let $S$ be the adjacency matrix of the graph, then
$$
A = \sum_{(i,j) \in E(G)} (1-y_i) y_j + (1-y_j) y_i = e^T S y - y^T S y
$$
where $e$ is the all-one vector and $y$ is the target $n$-dimensional $\{0,1\}$ vector. We can merge the coefficient of linear component of $y$ in $A$ with $B_i$ and get the final decision rule (ADMM):
$$
\begin{align}
\min\, & y^T S y- b^T y \\
s.t.\,\, & y \in \{0, 1\}^n \\
 &e^T y = \frac{n}{2}
\end{align}
$$
where $b_i = \textrm{deg}(i) + B_i$.

p(x, z | y_1)  is difficult to estimate, **sum** cannot be simplified.

We consider $p(y_1 | x, y_2, \dots, y_n ,z)$ instead. Assuming $y_i \sim Bern(0.5)$ That is, we do not require $e^t y = \frac{n}{2}$ exactly.

$H_0: p(x,z | y_2, \dots, y_n, y_1 = 0) = p_0(x) \prod_{i \in N_1(G)} p^{1-y_i} q^{y_i} \prod_{i \not \in N_1(G)} (1-p)^{1-y_i} (1-q)^{y_i} f(y_2, \dots, y_n)$

And

$H_1: p(x, z | y_2, \dots, y_n, y_1 = 1) = p_1(x) \prod_{i \in N_1(G)} p^{y_i} q^{1-y_i} \prod_{i \not \in N_1(G)} (1-p)^{y_i} (1-q)^{1-y_i} f(y_2, \dots, y_n)$

The decision rule to accept $H_0$ is $A(x, y_2, \dots, y_n, z) : = \{(x, y_2, \dots, y_n,z) | p(x,z | y_2, \dots, y_n, y_1 = 0) > p(x,z | y_2, \dots, y_n, y_1 = 1) \}$
$$
A(x, y_2, \dots, y_n, z) : =\{(x, y_2, \dots, y_n,z) | \sum_{i=1}^m \log \left(\frac{p_0(x_i^{(1)})}{p_1(x_i^{(1)})}\right) + \sum_{j=2}^n (1-2y_i)[z_{1j}\log\frac{p}{q} + (1-z_{1j})\log\frac{1-p}{1-q}] > 0 \}
$$

If $y_i = 0, i = 2, \dots, n$ then the decision rule can be written as:
$$
m [D(P_{X^m} || p_1) - D(P_{X^m} || p_0)] + (n-1) [D(P_Z || P_{Z_1}) - D(P_Z || P_{Z_0})] > 0
$$
where $P_{Z_1}\sim Bern(q)$, $P_{Z_0} \sim Bern(p)$ and $P_Z$ is a distribution on $\{0, 1\}$.

For general case, let $\kappa = \sum_{i=2}^n y_i$. Then we need two random variables on $\{0, 1\}$ : $Z, Z'$ such that

the decision rule (to accept $H_0$) is
$$
m [D(P_{X^m} || p_1) - D(P_{X^m} || p_0)] + (n-1 -\kappa) [D(P_Z || P_{Z_1}) - D(P_Z || P_{Z_0})] > \kappa [D(P_{Z'} || P_{Z_1}) - D(P_{Z'} || P_{Z_0})] 
$$





## Huang's suggestion:

We suppose the prior distributions for $y_1, \dots, y_n$ are i.i.d Bern(0.5). $y_2, \dots, y_n$ should be hidden, to compare $p(x, z | y_1 =1)$ with $p(x,z| y_1 = 0)$ we only need

to compare $\sum_{y_2, \dots, y_n} p(x,z,y_1=1, y_2, \dots, y_n)$ with $\sum_{y_2, \dots, y_n} p(x,z,y_1=0, y_2, \dots, y_n)$.

A suboptimal algorithm:

To estimate $Y_1$, we first estimate $Y_i$ from sample $\{X^{(i)}_1, \dots, X^{(i)}_{m_i}\} $

The posterior distribution for $Y_i$ is $\Pr(Y_i=0 | x^{(i)}_1, \dots, x^{(i)}_{m_i}) = \frac{\prod_{j=1}^{m_i} p_0(x^{(i)}_j)}{\prod_{j=1}^{m_i} p_0(x^{(i)}_j)+\prod_{j=1}^{m_i} p_1(x^{(i)}_j)} $
$$
\Pr(Y_i=0 | x^{(i)}_1, \dots, x^{(i)}_{m_i})= \frac{1}{1+\exp\left(-\displaystyle\sum_{j=1}^{m_i} \log \frac{p_0(x^{(i)}_j)}{p_1(x^{(i)}_j)}\right)}
$$
Suppose $x_1^{(i)}, \dots x_{m_i}^{(i)}$ is sampled from $p_0$. Then as $m_i \to \infty$, $\Pr(Y_i=0 | x^{(i)}_1, \dots, x^{(i)}_{m_i}) = \frac{1}{1+\exp^{-m_i D(p_0 || p_1)}} \sim 1 - \exp^{-m_i D(p_0 || p_1)}$. Therefore, the type I error is $\Pr(Y_i=1 | x^{(i)}_1, \dots, x^{(i)}_{m_i}) \sim \exp^{-m_iD(p_0 || p_1)}$

A naive upper bound for this 2-stage method: $\sum_{i=2}^n \exp^{-m_i D(p_0 || p_1)} + \exp^{-m_1 D(p_0 || p_1) - n D(p || q)}$



## Chernoff alpha Bound for SBM

[1] Zhou, Zhixin, and Ping Li. "Rate optimal Chernoff bound and application to community detection in the stochastic block models." *Electronic Journal of Statistics* 14.1 (2020): 1302-1347.

[2] Zhang, Anderson Y., and Harrison H. Zhou. "Minimax rates of community detection in stochastic block models." *The Annals of Statistics* 44.5 (2016): 2252-2280.

In paper [2], for a given $K$ community SBM,

the mismatched error $Er(\sigma, \hat{\sigma})$ for an estimator $\hat{\sigma}$:
$$
\lim_{n\to\infty}-\frac{1}{n}\log\inf_{\hat{\sigma}} Er(\sigma, \hat{\sigma}) = \frac{I}{K}
$$
where
$$
I = -2\log \left(\sqrt{\frac{a}{n}\frac{b}{n}} + \sqrt{(1-\frac{a}{n})(1-\frac{b}{n})}\right)
$$
which is the Renyi divergence of order $1/2$.

Notice that in this kind of literature $r(\sigma, \hat{\sigma})$ is a fraction between 0 and 1. This definition of error function

is different with that of Abbe, since Abbe focuses on the problem of exact recovery while the former focuses on the error proportion.

## Mixed Hypothesis Testing Problem

$H_0: (p_0, q_0)$ and $H_1: (p_1, q_1)$. $p_i$ is a distribution on $\mathcal{X}$ while $q_i$ is on $\mathcal{Z}$.

$X_1, \dots, X_m$ are generated from $p_i$ while $Z_1, \dots, Z_n$ are generated from $q_i$. Now given 

$X_1, \dots, X_m, Z_1, \dots, Z_n$ choose $i=0$ or $i=1$?

Using Log likelihood ratios test we have
$$
m \sum_{i=1}^m\log \frac{p_0(x_i)}{p_1(x_i)}+n \sum_{i=1}^n \log \frac{q_0(z_i)}{q_1(z_i)} > \log T
$$
where $T$ is a threshold.

Let the empirical distribution from the sample be $X_m, Z_n$ then the LRT is equivalent with
$$
A(P_{X^m}, P_{Z^n})=\{m (D(P_{X^m}||p_1) - D(P_{X^m} || p_0)) + n (D(P_{Z^n}||q_1) - D(P_{Z^n} || q_0))> \log T\}
$$
To analyze the problem, we first extend the "Probability of Type Class" to encompass a more general case. 

To estimate $P_0((P_X, P_Z)\in A^c)$ we have (See the Textbook, Elements of Information Theory Page 362)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         
$$
\begin{align}
P_0((P_X, P_Z)\in A^c) & = \sum_{(P_X, P_Z)\in A^c} P_0^m(T(P_X)) Q_0^n(T(P_Z))
\leq \sum_{(P_X, P_Z)\in A^c} \exp(-m D(P_X || P_0) - n D(P_Z || Q_0))
\end{align}
$$
Suppose $m=n$ ($m=kn$ can also be considered where $k$ is a constant). After some computation we can show that
$$
\lim_{n\to\infty}-\frac{1}{n} \log P_0((P_X, P_Z)\in A^c)  = D(P^*_{X} || P_0)  + D(P^*_Z || Q_0)
$$
where $(P_X^*, P_Z^*)$ is $\arg\min_{(P_X,P_Z) \in A^c}D(P_X|| P_0)+D(P_Z || Q_0)$

The optimal $\lambda$ is chosen such that $D(P_{\lambda} || P_0) + D(Q_{\lambda} || Q_0) = D(P_{\lambda} || P_1) + D(Q_{\lambda} || Q_1)$, where
$$
\begin{align}
P_{\lambda} & = \frac{P_0^{\lambda}P_1^{1-\lambda}}{\sum_{a\in \mathcal{X}} P_0^{\lambda}(a)P_1^{1-\lambda}(a)}\\
Q_{\lambda} & = \frac{Q_0^{\lambda}Q_1^{1-\lambda}}{\sum_{a\in \mathcal{Z}} Q_0^{\lambda}(a)Q_1^{1-\lambda}(a)}
\end{align}
$$

## When $p,q$ are constants

When $p>q$ and $p,q$ are constants, it is known that exact discovery is possible when $n\to\infty$.



This section is based on deduction by Jin Sima.

Suppose $y_1=1, y_2=0, y_3 = 1, y_4 = 0, \dots, y_{n-1}=1, y_n=0$ is the ground truth $A$.

Let $A_k$ be the event that maximum likelihood method gives an estimator which has $k$ pairs different with

the ground truth. The total error probability $P^{(e)}=\sum_{i=1}^{n/2}P(A_k)$.

For each specific $A_k$ the distinguished pair has $\binom{n/2}{k}^2$ number of choices.

Let $P_n^{(k)}$ to represent the error probability for the event when a specific choice has

larger probability than the ground truth. Then $P(A_k) = \binom{n/2}{k}^2 P_n^{(k)}$.

We consider $k=1$ first, which can be the lower bound of $P^{(e)}$.

We consider the specific choice:
$$
\begin{align}
A: y_1 = 1, y_2 = 0, y_3 = 1, y_4 = 0, \dots, y_{n-1} = 1, y_n = 0 \textrm{ ground truth} \\
A_1: y_1 = 0, y_2 = 1, y_3 = 1, y_4 = 0, \dots, y_{n-1} = 1, y_n = 0 \\
\end{align}
$$
which differs from $A$ at $y_1, y_2$.
$$
P(A) = \prod_{i=1}^m p_1(x_{1i})\prod_{i=1}^m p_0(x_{2i})\prod_{\substack{i=3\\i \textrm{ is odd}}}^n p^{z_{1i}}(1-p)^{1-z_{1i}}\prod_{\substack{i=3\\i \textrm{ is even}}}^n q^{z_{1i}}(1-q)^{1-z_{1i}}\prod_{\substack{i=3\\i \textrm{ is odd}}}^n q^{z_{2i}}(1-q)^{1-z_{2i}}\prod_{\substack{i=3\\i \textrm{ is even}}}^n p^{z_{2i}}(1-p)^{1-z_{2i}}
$$

$$
P(A_1) = \prod_{i=1}^m p_0(x_{1i})\prod_{i=1}^m p_1(x_{2i})\prod_{\substack{i=3\\i \textrm{ is odd}}}^n q^{z_{1i}}(1-q)^{1-z_{1i}}\prod_{\substack{i=3\\i \textrm{ is even}}}^n p^{z_{1i}}(1-p)^{1-z_{1i}}\prod_{\substack{i=3\\i \textrm{ is odd}}}^n p^{z_{2i}}(1-p)^{1-z_{2i}}\prod_{\substack{i=3\\i \textrm{ is even}}}^n q^{z_{2i}}(1-q)^{1-z_{2i}}
$$

Then

$P_n^{(1)} = P(P(A) < P(A_1))$

where
$$
P(A) < P(A_1) \Rightarrow 
m [D(X_1^m || P_1) - D(X_1^m || P_0)] + m [D(X_2^m || P_0) - D(X_2^m || P_1)]
> (n-2)[D(P_{Z_1^{n-2}}|| P_{Z_q})-D(P_{Z_1^{n-2}}|| P_{Z_p})] + (n-2)[D(P_{Z_2^{n-2}}|| P_{Z_p})-D(P_{Z_2^{n-2}}|| P_{Z_q})]
$$
Where $X_j^m$ is empirical distribution from the sample $x_{j1}, x_{j2}, \dots, x_{jm}$

$Z_1^{n-2}$ is empirical distribution from the sample $z_{13}, z_{15}, \dots, z_{1,n-1}, z_{24}, \dots, z_{2n} $; (Bern(p))

$Z_2^{n-2}$ is empirical distribution from the sample $z_{14}, z_{16}, \dots, z_{1,n}, z_{23}, \dots, z_{2,n-1} $; (Bern(q))

The decision rule  can also be written in the following form:
$$
\sum_{i=1}^m \log\frac{p_0(x_{1i})}{p_1(x_{1i})} +
\sum_{i=1}^m \log\frac{p_1(x_{2i})}{p_0(x_{2i})}
\geq \log \frac{p(1-q)}{q(1-p)} \sum_{i=1}^{n-2}(z_{i1} - z_{i2})
$$

For general $k$, the decision rule is:
$$
\sum_{i=1}^{km} \log\frac{p_0(x_{1i})}{p_1(x_{1i})} +
\sum_{i=1}^{km} \log\frac{p_1(x_{2i})}{p_0(x_{2i})}
\geq \log \frac{p(1-q)}{q(1-p)} \sum_{i=1}^{k(n-2k)}(z_{i1} - z_{i2})
$$
where $x_{1i}$ are sampled from $p_1$, $x_{2i}$ from $p_0$;

$z_{i1}$ are sampled from Bern(p), $z_{i2}$ sampled from Bern(q).

Let $m=n$, from Sanov's theorem, when $k\ll n$ we can show that $P(A_k) \dot{=} \exp(-kn C)$.

Therefore, the dominant term is $k=1$.

Notice that $\log \frac{p(1-q)}{q(1-p)}  > 0$. We consider the case when $m$ is very large, then by law of large number

we have $\sum_{i=1}^m \log\frac{p_0(x_{1i})}{p_1(x_{1i})} +
\sum_{i=1}^m \log\frac{p_1(x_{2i})}{p_0(x_{2i})} \to - (D(p_0 || p_1) + D(p_1 || p_0))$.

Let $D=\frac{D(p_0 || p_1) + D(p_1 || p_0)}{\log \frac{p(1-q)}{q(1-p)}}$.

Then we want to estimate the error term $P(\sum_{i=1}^{n-2} Z_{i2} - Z_{i1} > mD)$.

If $p = a \log n /n, q = b \log n /b $, then $D\to \frac{D(p_0 || p_1) + D(p_1 || p_0)}{\log \frac{a}{b}}$, which is a constant.

Therefore, we will treat $D$ as a constant to estimate the case of one error.

### Impact of $p(n),q(n)$

When $n$ is large, we can find $\epsilon_2$ such that $\log \frac{p(1-q)}{q(1-p)} \leq \log \frac{a}{b} + \epsilon_2$

By Sanov's theorem:
$$
P(\frac{1}{m}\sum_{i=1}^{m} \log \frac{p_1(x_{1i})}{p_0(x_{1i})} < D(p_1||p_0) - \frac{\epsilon_1}{2}) \dot{=} \exp(-m D(p_{\epsilon_1}^* || p_1))
$$
which decays in polynomial rate if $m=O(\log n)$.

Therefore we can conditioned $\log \frac{p(1-q)}{q(1-p)}\sum_{i=1}^{n-2} Z_{i2} - Z_{i1} > \sum_{i=1}^m \log\frac{p_1(x_{1i})}{p_0(x_{1i})} +
\sum_{i=1}^m \log\frac{p_0(x_{2i})}{p_1(x_{2i})}$ on
$$
\begin{align}
m\frac{1}{m}\sum_{i=1}^{m} \log \frac{p_1(x_{1i})}{p_0(x_{1i})} &\geq D(p_1||p_0) - \frac{\epsilon_1}{2}\\
m\frac{1}{m}\sum_{i=1}^{m} \log \frac{p_0(x_{2i})}{p_1(x_{2i})} &\geq D(p_0||p_1) - \frac{\epsilon_1}{2}\\
\end{align}
$$
Let $D(\epsilon_1, \epsilon_2) = \frac{D(p_0 || p_1) + D(p_1 || p_0) - \epsilon_1}{\log \frac{a}{b} + \epsilon_2}$.

Then 
$$
P(\log \frac{p(1-q)}{q(1-p)}\sum_{i=1}^{n-2} Z_{i2} - Z_{i1} > \sum_{i=1}^m \log\frac{p_1(x_{1i})}{p_0(x_{1i})} +
\sum_{i=1}^m \log\frac{p_0(x_{2i})}{p_1(x_{2i})}) < P(\sum_{i=1}^{n-2} Z_{i2} - Z_{i1} > mD(\epsilon_1, \epsilon_2))(1-\exp(-m C_1) - exp(-m C_2)) + \exp(-m C_1) + \exp(-m C_2)
$$




Therefore, from Sanov's theorem, asymptotically we have $P_n^{1}\asymp\exp(-m C_1 - (n-2)C_2)$

Similarly, for other $k$ we have $P_n^k \asymp\exp(-mk C_1 - k(n-2k)C_2)$.

The problem is that $C_2$ depends on $k$. This is because $C_2 = D(p^*_{1\lambda} || p) + D(p^*_{2\lambda} || q)$ while $\lambda$ is determined

from an equation containing $k$.



Actually, the dominate term is $P_n^1$. Our conclusion is that when $p, q$ are constant and $m=O(n)$, the

exact recovery error decreases in $\exp(-m C_1 - n C_2)$.

## Critical condition

When the graph is missing,  the exact recovery condition is $n^2 \exp(-m D)$ as $m\to \infty$.

Using the conclusion from "Mixed Hypothesis Testing Problem", we should choose $\lambda=\frac{1}{2}$

and $p_1^*=p_2^*$. Therefore $D=D(p_1^* || p_0) + D(p_1^* || p_1)$.

 $\to D=-2 \log \sum_{x\in \mathcal{X}} \sqrt{P_0(x)P_1(x)}$.

The value $D$ is called Renyi divergence of order $1/2$.

We let $m=\log n \to D > 2$ 



When the node information is missing, the condition is $n^2 \exp(-n D)$​ where

$ D = -2 \log \sum_{x\in \mathcal{X}} \sqrt{Q_0(x)Q_1(x)} = \frac{(\sqrt{a}-\sqrt{b})^2 \log n}{n}$

Therefore $\sqrt{a} - \sqrt{b} > \sqrt{2}$.



## Mixed Case

Using Lemma 7 of Abbe's paper [1],

the error exponent of $P(\sum_{i=1}^n (z_{i2} - z_{i1} \geq \epsilon \log n ))$ is $\exp(-\log n \cdot g(a, b, \epsilon))$

For a given type $T(p)$, let $m= \gamma\log n $ , we have
$$
\epsilon = -\gamma\frac{D(X_1^m || P_1) - D(X_1^m || P_0) + D(X_2^m || P_0) - D(X_2^m || P_1)}{\log a /b}
$$
From Theorem 11.1.4 of [2], the error exponent of a specific type is

$\exp(-m (D(X_1^m || p_1) + D(X_2^m || p_0)))$

To maximize  $\exp(-\log n g(a, b, \epsilon))\exp(-m (D(X_1^m || p_1) + D(X_2^m || p_0)))$ is equivalent to (unconstrained optimization):
$$
\theta^* = \min \gamma(D(X_1^m || p_1) + D(X_2^m || p_0)) + g(a, b, \epsilon)
$$
Then $P(P(A)<P(A_1))\leq \sum_{P\in P_n} Q^n(T(P))P(\sum_{i=1}^n (z_{i1} - z_{i2} \geq \epsilon \log n )) \leq |P_n| \exp(-\log n \cdot\theta^*)$.

Since $|P_n|\leq m^{|\mathcal{X}|^2} = O((\log n)^{|\mathcal{X}|^2})$, which is inferior to the polynomial term. Therefore, we get the

dominant error term as $n^{-\theta^*}$.

To get the lower bound, we only need to choose a $P^*$ to reach $\theta^*$ and we then have

$P(P(A) > P(A_1)) \geq Q^n(T(P^*))P(\sum_{i=1}^n (z_{i1} - z_{i2} \geq \epsilon \log n )) \geq \exp(-\log n \cdot \theta^*)$.

Here we also use the conclusion that $\frac{1}{(m+1)^{|\mathcal{X}|^2}}2^{-mD(P||Q)}\leq Q^n(T(P))$ and neglect the inferior term.



Actually we can show that $\frac{d^2 g(a,b,\epsilon) }{d\epsilon^2} > 0$ therefore
$$
\gamma(D(X_1^m || p_1) + D(X_2^m || p_0)) + g(a, b, \epsilon) \geq \gamma(D(X_1^m || p_1) + D(X_2^m || p_0))  + \tilde{g}(a,b,\epsilon)
$$


where
$$
\tilde{g}(a, b, \epsilon) = (\sqrt{a} - \sqrt{b})^2 + \frac{\epsilon}{2}\log \frac{a}{b}
$$
The coefficient of $\epsilon > 0$.

Now we minimize $\gamma(D(X_1^m || p_1) + D(X_2^m || p_0))  + \tilde{g}(a,b,\epsilon)$

Let $\kappa = \frac{1}{2}$.

we can get the optimal  distribution for $X_1^m, X_2^m$:
$$
\begin{align}
P_{X_1^m}(x) &= \frac{P_1^{1-\kappa}(x)P_0^\kappa(x)}{\sum_{x\in\mathcal{X}}P_1^{1-\kappa}(x)P_0^\kappa(x)} \\
P_{X_2^m}(x) &= \frac{P_0^{1-\kappa}(x)P_1^\kappa(x)}{\sum_{x\in\mathcal{X}}P_0^{1-\kappa}(x)P_1^\kappa(x)}
\end{align}
$$
Take $X_1^m$ for an example, we have $P_{X_1^m}^* = \arg\min [(1-\kappa)D(X_1^m || p_1) + \kappa D(X_1^m || p_0)]$

And the optimal value is
$$
\Gamma(a,b,p_0, p_1) = (\sqrt{a} -\sqrt{b})^2 - 2\gamma\log(\sum_{x\in\mathcal{X}} \sqrt{p_1(x)p_0(x)})
$$
Notice that $\Gamma(a,b,p_0, p_1) $ is linear with $\gamma$.

The term $-\log(\sum_{x\in\mathcal{X}} p_0^{1-\kappa}(x)p_1^{\kappa}(x))$ is related with Renyi divergence of order $\kappa$
$$
D_{\kappa}(p_1 || p_0) = -\frac{1}{1-\kappa}\log(\sum_{x\in\mathcal{X}} p_0^{1-\kappa}(x)p_1^{\kappa}(x))
$$
The exact recovery condition is:
$$
\begin{align}
\Gamma(a,b,p_0, p_1) &> 2 \\
\epsilon &> b - a
\end{align}
$$

Let $C=\sum_{x\in\mathcal{X}} p^{1-\kappa}_1(x)p^{\kappa}_0(x), C'=\sum_{x\in\mathcal{X}} p^{1-\kappa}_0(x)p^{\kappa}_1(x)$

The second condition is equivalent with:
$$
h(\kappa)=\frac{1}{C}\sum_{x\in\mathcal{X}} p^{1-\kappa}_1(x)p^{\kappa}_0(x)\log\frac{p_0(x)}{p_1(x)}+\frac{1}{C'}\sum_{x\in\mathcal{X}} p^{1-\kappa}_0(x)p^{\kappa}_1(x)\log\frac{p_1(x)}{p_0(x)} < \frac{a-b}{\gamma}\log\frac{a}{b}
$$
Since $\kappa = \frac{1}{2}$, the right hand side is equal to 0 ($h(\frac{1}{2})=0, h(0)<0, h(1)>0$). Therefore the second condition holds.



## Consideration for general $k$

$$
\epsilon = -\gamma k\frac{D(X_1^{km} || P_1) - D(X_1^{km} || P_0) + D(X_2^{km} || P_0) - D(X_2^{km} || P_1)}{\log a /b}
$$

and we minimize
$$
k\gamma (D(X_1^{km}||P_1) + D(X_2^{km}||P_0)) + \frac{k(n-2k)}{n} g(\alpha, \beta, \frac{n}{k(n-2k)}\epsilon)
$$
Where the function $g(\alpha, \beta, \epsilon)$ is defined as:
$$
g(\alpha, \beta, \epsilon) = \begin{cases}
0 & \epsilon < (b-a) \\
 (\alpha+\beta)-\epsilon \log(\beta) - 2\sqrt{\left( \frac{\epsilon}{2}\right)^2 +\alpha \beta} +\frac{\epsilon}{2} \log \left( \alpha \beta \frac{\sqrt{(\epsilon/2)^2 +\alpha \beta} +\epsilon/2}{\sqrt{(\epsilon/2)^2 +\alpha \beta} -\epsilon/2} \right)& |\epsilon| \leq(a-b) \\
+\infty & \epsilon > (a-b)
\end{cases}
$$

We can use Lagrange multiplier to solve this problem:

We add a term $-\lambda[ \gamma k[D(X_1^{km} || P_1) - D(X_1^{km} || P_0) + D(X_2^{km} || P_0) - D(X_2^{km} || P_1)]+ \log\frac{a}{b}\epsilon]$

Then $g'(\epsilon') = \lambda \log\frac{a}{b}$. where $\epsilon' = \frac{n}{k(n-2k)}\epsilon$

Let $\lambda = \frac{1}{\log(a/b)}\log\frac{\epsilon'+\sqrt{\epsilon'^2+4ab}}{2b}$.

Then $h(\lambda) = -\frac{\log(a/b)\epsilon}{\gamma k}$ from which we can solve $\epsilon(k)=0$.

Then the minimum value equals
$$
\Gamma(k) = \frac{k(n-2k)}{n}[a+b-\sqrt{\epsilon'^2 + 4ab}] - k\gamma(\log C + \log C')
$$
Let $H(\lambda) = \log C + \log C'$, then $H'(\lambda) = h(\lambda)$
$$
h(\lambda)=\frac{1}{C}\sum_{x\in\mathcal{X}} p^{1-\lambda}_1(x)p^{\lambda}_0(x)\log\frac{p_0(x)}{p_1(x)}+\frac{1}{C'}\sum_{x\in\mathcal{X}} p^{1-\lambda}_0(x)p^{\lambda}_1(x)\log\frac{p_1(x)}{p_0(x)}
$$
We transform the expression to $\lambda$, we have
$$
\begin{align}
\epsilon' &= b \exp(\lambda \log(a/b)) - a\exp(-\lambda  \log(a/b)) \\
\sqrt{\epsilon'^2 + 4ab}  &= b \exp(\lambda  \log(a/b)) + a\exp(-\lambda  \log(a/b)) \\
\end{align}
$$
Then
$$
\Gamma(k) = \frac{k(n-2k)}{n}(a+b) - \Xi[\lambda]
$$
where
$$
\Xi[\lambda] = \frac{k(n-2k)}{n}(b \exp(\lambda  \log(a/b)) + a\exp(-\lambda \log(a/b))) + \gamma k H(\lambda)
$$
$\Xi'[\lambda] = \log(a/b) \epsilon + \gamma k h(\lambda) = 0$.

$\Xi''[\lambda] = \frac{k(n-2k)}{n}\log^2(a/b)(b \exp( \lambda\log(a/b)) + a\exp(-\lambda \log(a/b))) + \gamma k h'(\lambda)>0$

Since
$$
h'(\lambda)=\frac{1}{C^2}(C\sum p_1^{1-\lambda}p_0^{\lambda}(\log\frac{p_0}{p_1})^2-(\sum p_1^{1-\lambda}p_0^{\lambda}\log\frac{p_0}{p_1})^2) + \frac{1}{C'^2}(C'\sum p_0^{1-\lambda}p_1^{\lambda}(\log\frac{p_1}{p_0})^2-(\sum p_0^{1-\lambda}p_1^{\lambda}\log\frac{p_1}{p_0})^2)
$$
By Cauchy Inequality, $h'(\lambda) > 0$.

Therefore, $\Gamma(k) = \frac{k(n-2k)}{n}(a+b) - \Xi[\lambda^*]$ where $\lambda^*$ is the minimum value of $\Xi$.

We have $\Xi[\lambda^*] \leq \Xi[\frac{1}{2}]$.

Therefore, $\Gamma(k) \geq  \frac{k(n-2k)}{n}(a+b) - \Xi[\frac{1}{2}]$.

We get a lower bound of the minimum value as:
$$
\Gamma(k) \geq \frac{k(n-2k)}{n}[a+b-2\sqrt{ab}] - 2\gamma k\log(\sum_{x\in\mathcal{X}}\sqrt{p_0(x)p_1(x)}) =:\Lambda(k)
$$
We then analyze the lower bound $\Lambda(k)$.



### When only one node has error

If the probability
$$
P(\sum_{i=1}^m \log\frac{p_0(x_{1i})}{p_1(x_{1i})} 
\geq \log \frac{p(1-q)}{q(1-p)} \sum_{i=1}^{n/2}(z_{i1} - z_{i2}))
$$
is considered,

we need to consider
$$
\theta^* = \min D(X_1^{m}||P_1)+  \frac{1}{2}g(a, b, 2\epsilon)
$$
And the probability is bounded by $\exp(-\log n \theta^*)$. (We suppose $m=\log n$ for simplicity).

We can get a lower bound of $\theta^*$ by considering only linear approximation of $\theta^*$.

Since $g(a,b,\epsilon) \geq (\sqrt{a} -\sqrt{b})^2 + \epsilon \frac{\log a/b}{2}$.

We consider
$$
\theta'=\min D(X_1^{m} || P_1) + \frac{1}{2}(\sqrt{a}-\sqrt{b})^2 + \epsilon * \log(a/b)/2
$$
where
$$
\epsilon = -\frac{D(X_1^{m}||P_1) - D(X_1^m || P_0)}{\log a/b}
$$
This is equivalent to minimize
$$
\frac{D(X_1^m || P_1) + D(X_1^m || P_0)}{2}
$$
And we get $\theta' = \frac{1}{2}(\sqrt{a}-\sqrt{b})^2 - \log\sum_{x\in \mathcal{X}}\sqrt{P_0(x)P_1(x)}$

That is we have
$$
P(\sum_{i=1}^m \log\frac{p_0(x_{1i})}{p_1(x_{1i})} 
\geq \log \frac{p(1-q)}{q(1-p)} \sum_{i=1}^{n/2}(z_{i1} - z_{i2}))\leq n^{-\frac{1}{2}(\sqrt{a}-\sqrt{b})^2 + \log\sum_{x\in \mathcal{X}}\sqrt{P_0(x)P_1(x)}}
$$


[1] Abbe, Emmanuel, Afonso S. Bandeira, and Georgina Hall. "Exact recovery in the stochastic block model." *IEEE Transactions on Information Theory* 62.1 (2015): 471-487.

[2] Cover, Thomas M. *Elements of information theory*. John Wiley & Sons, 1999.
Consider $k=2$, where $B_i \sim Binom(\frac{n}{2}, b\frac{\log n}{n})$

 and $A_i \sim Binom(\frac{n}{2}, a\frac{\log n}{n})$



Unconditional expectation:
$$
E[\sum_{i=1}^n \exp(\beta(B_i - A_i))] = (1-\frac{\log^2 n}{4n}(a^2(e^{-\beta}-1)^2+b^2(e^{\beta}-1)^2))n^{g(\beta)}
$$
Therefore
$$
\frac{P_{\sigma |G}(\mathrm{dist}(\sigma,X)=1)}{P_{\sigma |G}(\sigma=X)}
= \exp(-\beta \gamma \frac{\log n}{n} )\sum_{i=1}^n \exp(\beta_n (B_i - A_i))
$$
where
$$
\beta_n = \beta(1+ \gamma \frac{\log n}{n})
$$


Adding connection with maximum likelihood model.



Let $\mathcal{G} = \{ B_i - A_i < 0, i=1,\dots, n \}$

Conditional expectation (does it work for $\beta > \beta^*$ ?)
$$
E[\sum_{i=1}^n \exp(\beta(B_i - A_i)) | G \in \mathcal{G}] = (1+o(1))n^{g(\beta)}
$$



We conduct an experiment to verify multiple samples $m>1$ and $k=2$:

We choose $n=3200, a=16, b=4, m=3$.

the empirical beta $\hat{\beta} = 0.0999$ while the theoretical transition point is $\beta^*/2 = 0.0991$
$$
\beta^*= \log(\frac{a + b -2 - \sqrt{(a + b - 2)^2 - 4 a b)}}{2  b})
$$


The acc versus $\beta$ near 0.1 is shown in the following figure (the red point is the empirical transition point):

![](./beta_trans-2020-11-14.svg)


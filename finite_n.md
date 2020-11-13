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

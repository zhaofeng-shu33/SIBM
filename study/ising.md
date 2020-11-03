## Application of Ising model

### Denoising images

https://pchanda.github.io/IsingModelDenoising/ has a good introduction on this topic.

Generally speaking, the true binary image is modeled by Ising model such that adjacent pixels

are more likely to take the same value. The noisy images are modeled as independent Gaussian

samples given the true but hidden image. The MAP estimator is to compute $P(X|Y=y)$ given the noise

sample $y$.  Usually we can get a good approximation $P(X|Y=y)$ by Gibb's sampling.


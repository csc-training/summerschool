# Calculating digits of π with Bailey-Borwein-Plouffe formula adn the Spigot algorithm

The code in `bbp-pi.cpp` calculates digits of π using the Bailey-Borwein-Ploueffe (BBP) series

$$\pi = \lim_{N\rightarrow \infty} S_N = \lim_{N\rightarrow\infty}\sum_{k=0}^N \left[ \frac 1 {16^k} \left ( \frac 4 {8k+1} - \frac 2 {8k+4} - \frac 1 {8k+5} - \frac 1 {8k+6} \right ) \right ]$$

and the [Spigot algorithm](https://en.wikipedia.org/wiki/Spigot_algorithm).

The idea is to scale $S_N$ by $16^{k}$ and look at the digits near the zero to
extract digits from this series.

The code is built upon the following ideas outlined below, but it might have
some room for improvement. See comments in the code for ideas. 

The (*open-ended*) task is to optimize the code in `bbp-pi.cpp`. The theory
below is not necessary to understand to be able to optimize the code.

## Background: Spigot algorithm

The spigot algorithm is used to calculate decimals of certain quickly
converging serie. It rests upon the following observations.

Denote fractional part of a number $s$ by $f(s)$, then

1. The fractional part of an rational number $\frac a b$ is given by $f(\frac {a}{b}) = \frac{a\mod b}{b}$.
2. The fractional part operation satisfies $$f\left(\sum_{k} \frac {a_k}{b_k}\right) = f\left(\sum_k f\left(\frac {a_k} {b_k}\right)\right).$$

We require the fractional part to be positive. Thus, for example,
$-16/5 = -(3\frac 1 5)= -4+4/5$, so fractional part of $-3.2$ is $0.8$. This
way we can add up positive and negative numbers in the above sum and the
observation 2. holds.

Note also that $f(k \frac a b) = f(k f(\frac a b))$ with $k\in \mathbb Z$.

Divide the series $S_N$ in question in head and tail parts $S_N = S_k + S_N-S_k = H_k + T_k^N$. Assume
that beyond certain $M>k$, when $N>M$ the first few digits of $T_k^N$ stay
fixed. Then using the above observations, the fractional part of the sum of the
head part and the truncated tail multiplied by a fitting large number will contain
the desired digits.

## Digits of π with the Spigot algorithm

Let us now denote $q_k^1 = 8k+1$, $q_k^2 = 8k+4$, $q_k^3 =8k+5$ and $q_k^4 = 8k+6$, and also
$p^1 = 4$, $p^2=-2$, $p^3=-1$ and $p^4=-1$ corresponding to the terms in the BBP series.

Now the fractional part of $16^d H_k$ is $f(\sum_{l=1}^k 16^{k-l}(\sum_{j=1}^4 \frac {p^j} {q^j_k}))$. 
Apply the above two observations on this sum and we get
$$f(H_k) = f(\sum_{j=1}^4 f(p^j f(\sum_{l=1}^k \frac {16^{k-l}\mod q^j_k}{q^j_k})))$$
and finally
$$f(H_k) = f(\sum_{j=1}^4 p^j \sum_{l=1}^k \frac {16^{k-l}\mod q^j_k}{q^j_k}).$$

For the tail part, we can just calculate couple terms of $16^d T_k^N$ since it converges quickly.

Now $$\lfloor16 f(16^d S_N)\rfloor = \lfloor 16 \left[f(16^d H_d) + f(16^d
T_d^N)\right]\rfloor$$ is the $d$th digit of π in hexadecimal.

**Caveat**: Calculating $16^{k-l} \mod q_k^j$ must be performed quickly.

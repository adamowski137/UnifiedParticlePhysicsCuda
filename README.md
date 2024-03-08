# UnifiedParticlePhysicsCuda

Project under development :))

## Overview ##
A project thats main goal is to simulate behavior of various bodies with the help of constrained based physics. The project is written such that the most operations are executed on GPU using CUDA technology.

Each body is divided into small particles that behavior is modeled by placing on them various constrains. If a constrain is broken by a particle it generates a force to negate it.

## Particle ##
A particle can be visualised as a struct:
```c++
struct particle
{
    float x, y, z;
    float vx, vy, vz;
    float invmass;
};
```
Where x, y and z indicate the particle position and vx, vy and vz indicate velocity. Many formulas require us to divide by mass so to prevent redundant operations we store its inverse.

## Constrain ##
A constrain is an object in the form of one of the following:
$$ 
    C(x_1, x_2 ... x_n) = 0
$$
$$
    C(x_1, x_2 ... x_n) \geq 0
$$

A constrain consists of the following:

- A cardinality $ n $ 
- A set of particles $ \{ x_1, x_2 ... x_n \} $
- A function $ C : \R^{3n} \rightarrow \R$

## Generating forces ##

Let's supose we have $ n $ points and $ c $ constrains. To generate forces we need to calculate how each constrain influences each particle. So we generate a matrix $ J \in \R^{3n \times c}$ the following way:
For particle $ p_i $ and constrain $ C_j $ we calculate $ \frac{ \partial C_j }{ \partial p_i }$ and fill $(3i, j), (3i + 1, j), (3i + 2, j)$ cells with calculated x, y and z coordinates. 
Simmilarly we generate the $ \dot{J} $ matrix where we instead use $ \frac{ \partial^2 C_j }{ \partial p_i \partial t }$. We need to generate a $ M^{-1} \in \R^{3n\times 3n}$ matrix which is just a diagonal matrix such that $ a_{i, i} = \frac{1}{m_{i/3}}$
After we generated all the needed matrices we can apply them to the following equation
$$
    JM^{-1}J^{T}\lambda = -\dot{J}\dot{q} - \alpha C - \beta \dot C
$$
where $ \alpha $ and $ \beta $ are magic constants. We solve this equation for $ \lambda $ using a iterative linear equations system soling method in our case it was the Jacobbi method.
The final step is to calculate the forces using 
$$ F = \lambda J^T$$

## Force aplication ##

We can then apply the forces to particles, update its velocity and position.
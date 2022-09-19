## SAC-LIKE Strategy FOR StocketMarket

Our goal is to apply RL to quantitative trading, enabling intelligent agents to complete trades in that complex environment.

For the current situation, we have constructed an extremely general system that allows existing RL algorithms to operate in that environment.

On the single agent side, many algorithms are designed to solve the MDP problem in RL, but the quantitative trading market, as an evolving scenario all the time, satisfies and constructs the POMDP problem.

Moreover, it has been shown in past RL studies that the benefits of addressing the POMDP issue are higher than those of addressing the MDP issue. Therefore, previous approaches in this direction by solving the MDP problem.

1). We innovatively change the problem to be solved from MDP to POMDP.

2). In the framework of maximum entropy, we optimise the choice of operations of the algorithm through entropy reduction operations such as knowledge structuring, thus making the gains more stable and better.

### Version changes

Two versions we have.

V1 doesn't have knowledge structuring and we can view it as a first edition.

V2 has a large change in actions selection. Such as we make the past half of a year data as a guide.
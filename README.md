Deterministic Distributional Reinforcement Learning (DDRL) combines deterministic approaches to RL [@barfuss2019deterministic] with distributional RL [@bellemare2017]. Within this framework, the learning dynamics of multi-agent games can be modeled and visualized as deterministic trajectories. DDRL also enables the implementation of cognitive heuristics (such as optimism/pessimism) as distortions of the value distributions. Therefore, it is particularly suited to studying the impact of cognitive heuristics on collective learning dynamics in a wide range of scenarios.

The present package and supplementary information were created thanks to [nbdev](https://nbdev.fast.ai/) and [quarto](https://quarto.org/). This ensures all the simulations are fully reproducible.  

The package and notebooks associated with this documentation can be found on [this Github repo](https://github.com/gerardoparra/pyDDRL). There, you can download the package and run the notebooks in order to reproduce our simulations.

To do this, clone the repository:

```sh
git clone https://github.com/gerardoparra/pyDDRL
cd pyDDRL
```

Then set up the virtual environment:

```sh
python -m venv .venv
source .venv/bin/activate
```

...and install the package in editable mode:

```sh
pip install -e .
``` 

Then you should be able to execute all the notebooks.

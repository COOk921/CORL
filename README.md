## Training OR model with PPO

### Container

python ppo_or.py --num-steps 51 --env-id container-v0 --env-entry-point envs.container_vector_env:ContainerVectorEnv --problem container

### TSP

```shell
python ppo_or.py --num-steps 51 --env-id tsp-v0 --env-entry-point envs.tsp_vector_env:TSPVectorEnv --problem tsp
```

### CVRP

```shell
python ppo_or.py --num-steps 60 --env-id cvrp-v0 --env-entry-point envs.cvrp_vector_env:CVRPVectorEnv --problem cvrp
```

### Enable WandB

```shell
python ppo_or.py ... --track
```


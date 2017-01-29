import numpy as np
from rllab.misc import ext
from rllab.sampler.parallel_sampler import _get_scoped_G


def do_training(g_opt_info, path):
    f_train_vfunc = g_opt_info.value["f_train_vfunc"]
    f_train_policy = g_opt_info.value["f_train_policy"]

    observations, actions, rewards, advantages = ext.extract(
        path,
        'observations', 'actions', 'rewards', 'advantages'
    )

    f_train_policy(observations, actions, advantages)
    f_train_vfunc(observations, advantages)


def train_worker(G, g_counter, g_opt_info, t_max, discount, lock, scope=None):
    G = _get_scoped_G(G, scope=scope)
    env = G.env
    f_train_vfunc = g_opt_info.value["f_train_vfunc"]
    f_train_policy = g_opt_info.value["f_train_policy"]
    policy = g_opt_info.value["target_policy"]
    vfunc = g_opt_info.value["target_vfunc"]

    obs = env.reset()
    observations = []
    actions = []
    rewards = []
    paths = []
    done = False
    while True:
        for _ in range(t_max):
            if done:
                env.reset()
            action, _ = policy.get_action(obs)
            next_obs, reward, done, info = env.step(action)
            observations.append(obs)
            actions.append(action)
            rewards.append(reward)
            obs = next_obs
            with lock:
                g_counter.value += 1
            if done:
                break

        # make it not expanding
        observations = observations[-t_max:]
        actions = actions[-t_max:]
        rewards = rewards[-t_max:]

        path = dict(
            observations=np.array(observations),
            actions=np.array(actions),
            rewards=np.array(rewards)
        )

        advantages = []
        returns = []
        if done: return_so_far = 0
        else:    return_so_far = vfunc.get_vval(obs)
        for t in range(len(rewards) - 2, -1, -1):
            return_so_far = rewards[t] + discount * return_so_far
            returns.append(return_so_far)
            advantage = return_so_far - path_baseline[t]
            advantages.append(advantage)
        # The advantages are stored backwards in time, so we need to revert it
        advantages = np.array(advantages[::-1])
        # The returns are stored backwards in time, so we need to revert it
        returns = returns[::-1]
        # normalize advantages
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)

        path["advantages"] = advantages
        path["returns"] = returns
        paths.apend(path)

        do_training(g_opt_info, path)





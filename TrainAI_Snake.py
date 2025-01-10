import os
import ray
import torch
import numpy as np
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from Game_component.snake_env import SnakeEnv



def env_creator(env_config):
    width = env_config.get("width", 20)
    height = env_config.get("height", 20)
    render = env_config.get("render", False)  # Отримуємо параметр render
    return SnakeEnv(width=width, height=height, render=render)  # Передаємо render


if __name__ == "__main__":
    ray.init(ignore_reinit_error=True)
    register_env("SnakeEnv-v1", env_creator)

    
    test_env = SnakeEnv(width=20, height=20)

    config = (
        PPOConfig()
        .api_stack(
            enable_rl_module_and_learner=True,
            enable_env_runner_and_connector_v2=True,
        )
        .environment(
            env="SnakeEnv-v1", 
            env_config={"width": 10, "height": 10, "render": False}
        )
        .env_runners(num_env_runners=14)
        .rl_module(
            model_config={
                "conv_filters": [
                    [16, [3, 3], 1],
                    [32, [3, 3], 1],
                ],
                "fcnet_hiddens": [2048, 1024, 512, 256, 256],
                "fcnet_activation": "relu",
                "action_distribution_cls": "Categorical"
            }
        )
        .training(
            lr=0.0001,
            gamma=0.99,
            #entropy_coeff=0.01,
            train_batch_size_per_learner=5000,  # Використовуємо новий параметр
            minibatch_size=500,
            num_epochs=10,
        )
        .framework("torch")
        # Налаштовуємо кількість графічних процесорів для Learner)
        .learners(num_gpus_per_learner=1)
    )


    # PPO
    #checkpoint_path = "D:\\python_project\\Snake_AI\\checkpoints_v8"
    algo = config.build()
    #algo.restore(checkpoint_path)

    num_iterations = 8000
    for i in range(num_iterations):
        result = algo.train()

        
        env_runners = result.get("env_runners", {})
        learners = result.get("learners", {})
        avg_reward = env_runners.get("episode_return_mean", None)


        # Отримуємо метрики з env_runners
        episode_return_mean = env_runners.get("episode_return_mean", None)
        episode_return_min = env_runners.get("episode_return_min", None)
        episode_return_max = env_runners.get("episode_return_max", None)
        episode_len_mean = env_runners.get("episode_len_mean", None)
        episode_len_min = env_runners.get("episode_len_min", None)
        episode_len_max = env_runners.get("episode_len_max", None)

        # Отримуємо метрики з learners
        policy_metrics = learners.get("default_policy", {})
        policy_loss = policy_metrics.get("policy_loss", None)
        vf_loss = policy_metrics.get("vf_loss", None)
        entropy = policy_metrics.get("entropy", None)
        mean_kl_loss = policy_metrics.get("mean_kl_loss", None)
        vf_explained_var = policy_metrics.get("vf_explained_var", None)
        gradients_norm = policy_metrics.get("gradients_default_optimizer_global_norm", None)
        learning_rate = policy_metrics.get("default_optimizer_learning_rate", None)

        num_episodes = env_runners.get("num_episodes", None)
        if num_episodes is not None:
            print(f"  Episodes this iteration: {num_episodes}")

        if avg_reward is not None:
            print(f"Iteration: {i+1},\n average reward: {avg_reward}")
            print(f"  Min: {episode_return_min}")
            print(f"  Max: {episode_return_max}")
            print(f"  Episode len mean: {episode_len_mean}")
            print(f"  Episode len min: {episode_len_min}")
            print(f"  Episode len max: {episode_len_max}")
            print(f"  policy loss: {policy_loss}")
            print(f"  vf loss: {vf_loss}")
            print(f"  entropy: {entropy}")
            print(f"  mean_kl_loss: {mean_kl_loss}")
            print(f"  vf_explained_var: {vf_explained_var}")
            print(f"  gradients_norm: {gradients_norm}")
            print(f"  learning_rate: {learning_rate}\n")
        else:
            print(f"Iteration: {i+1}, average reward: Metric not found. ALL result: {result}")



        
        # Зберігаємо контрольні точки кожні 500 ітерацій
        if (i + 1) % 500 == 0:
            # Перевіримо чи існує папка 'checkpoints', якщо ні — створимо
            checkpoints_dir = "checkpoints_v8"
            if not os.path.exists(checkpoints_dir):
                os.makedirs(checkpoints_dir)

            checkpoint_dir = algo.save("D:\\python_project\\Snake_AI\\checkpoints_v8")
            print(f"The checkpoint is saved in: {checkpoint_dir}")



        # Кожні 100 ітерацій виконати повну симуляцію епізоду з візуалізацією
        if (i + 1) % 100 == 0:
            eval_env = SnakeEnv(width=10, height=10, render=True)
            obs, info = eval_env.reset()
            done = False
            truncated = False
            total_reward = 0.0

            # Отримуємо модуль політики
            module = algo.get_module("default_policy")

            # Рендеримо початковий стан
            eval_env.render()

            # Лічильник кроків у епізоді
            step_count = 0
            max_steps = 2000

            # Повний епізод
            while not done and not truncated:
                # Якщо перевищили 2000 кроків - виходимо
                if step_count > max_steps:
                    print(f"Episode exceeded {max_steps} steps, terminating early.")
                    break

                # Використовуємо forward_inference для отримання логітів дій
                obs_batch = torch.from_numpy(np.array([obs], dtype=np.float32))
                inference_out = module.forward_inference({"obs": obs_batch})
                action_logits = inference_out["action_dist_inputs"]

                # Вибираємо дію як argmax
                action = torch.argmax(action_logits[0]).item()

                obs, reward, done, truncated, info = eval_env.step(action)
                total_reward += reward
                eval_env.render()

                step_count += 1

            print(f"Visualization episode after iteration {i+1}, total reward: {total_reward}")
            eval_env.close()



    # Завершення роботи Ray
    ray.shutdown()

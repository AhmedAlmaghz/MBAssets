#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TensorFlow الاحترافي — التعلم المعزز
Professional TensorFlow — Rl

Book slug : tensorflow-professional-ar
Chapter   : ch-21-ch21-rl (lesson 21)
Code files: 5 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/tensorflow-professional-ar/code/ch-21-ch21-rl.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TensorFlow الاحترافي — التعلم المعزز
Professional TensorFlow — Rl

Book slug : tensorflow-professional-ar
Chapter   : ch-21-ch21-rl (lesson 21)
Code files: 4 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book/src/chapters/ch21_rl.md
# ────────────────────────────────────────────────────────────────────────
target = reward + gamma * max_next_q
new_q = old_q + learning_rate * (target - old_q)

# ────────────────────────────────────────────────────────────────────────
# [002] book/src/chapters/ch21_rl.md
# ────────────────────────────────────────────────────────────────────────
if rng.random() < epsilon:
    action = env.action_space.sample()
else:
    action = int(q_values.argmax())

# ────────────────────────────────────────────────────────────────────────
# [003] book/src/chapters/ch21_rl.md
# ────────────────────────────────────────────────────────────────────────
import tensorflow as tf

q_values = online_network(states)
chosen_q = tf.reduce_sum(q_values * tf.one_hot(actions, num_actions), axis=1)
next_q = tf.reduce_max(target_network(next_states), axis=1)
target = rewards + gamma * (1.0 - terminals) * next_q
loss = tf.reduce_mean(tf.square(tf.stop_gradient(target) - chosen_q))

# ────────────────────────────────────────────────────────────────────────
# [004] book/src/chapters/ch21_rl.md
# ────────────────────────────────────────────────────────────────────────
returns = []
for episode in range(5):
    state, _ = env.reset(seed=100 + episode)
    returns.append(run_episode(policy, env, state))
print(sum(returns) / len(returns))

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/tensorflow-professional-ar/notebooks/ch-21-ch21-rl.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
target = reward + gamma * max_next_q
new_q = old_q + learning_rate * (target - old_q)

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/tensorflow-professional-ar/notebooks/ch-21-ch21-rl.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
if rng.random() < epsilon:
    action = env.action_space.sample()
else:
    action = int(q_values.argmax())

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/tensorflow-professional-ar/notebooks/ch-21-ch21-rl.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import tensorflow as tf

q_values = online_network(states)
chosen_q = tf.reduce_sum(q_values * tf.one_hot(actions, num_actions), axis=1)
next_q = tf.reduce_max(target_network(next_states), axis=1)
target = rewards + gamma * (1.0 - terminals) * next_q
loss = tf.reduce_mean(tf.square(tf.stop_gradient(target) - chosen_q))

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/tensorflow-professional-ar/notebooks/ch-21-ch21-rl.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
returns = []
for episode in range(5):
    state, _ = env.reset(seed=100 + episode)
    returns.append(run_episode(policy, env, state))
print(sum(returns) / len(returns))

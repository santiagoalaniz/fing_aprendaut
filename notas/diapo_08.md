# Aprendizaje por refuerzo

https://www.youtube.com/watch?v=AhyznRSDjw8

Reinforcement learning es un campo que se centra en enseñar a agentes autónomos cómo tomar acciones óptimas en un entorno para lograr sus objetivos. Estos agentes pueden aprender a realizar tareas como controlar robots, optimizar operaciones en fábricas o jugar juegos de mesa. 

Un entrenador puede ofrecer recompensas o penalizaciones basadas en las acciones del agente para guiar su aprendizaje. El agente aprende a elegir secuencias de acciones que maximicen la recompensa acumulada a lo largo del tiempo. Un algoritmo clave en este campo es el Q-learning, que permite al agente aprender estrategias óptimas incluso sin conocimiento previo del entorno. 

Estos algoritmos están relacionados con los algoritmos de programación dinámica utilizados en problemas de optimización.

## Resumen

Este capítulo se enfoca en cómo los agentes autónomos pueden aprender políticas de control exitosas mediante la experimentación en su entorno. Se asume que los objetivos del agente pueden definirse mediante una función de recompensa que asigna un valor numérico o "pago inmediato" a cada acción distinta que el agente pueda tomar desde cada estado distinto. 

Por ejemplo, el objetivo de conectarse a un cargador de batería podría reflejarse asignando una recompensa positiva a las transiciones de estado-acción que resulten en una conexión inmediata al cargador, y una recompensa de cero para todas las demás. Esta función de recompensa podría estar incorporada en el robot o ser proporcionada por un maestro externo. 

La tarea del robot es realizar secuencias de acciones, observar sus consecuencias y aprender una política de control. El objetivo es que esta política permita al agente maximizar la recompensa acumulada con el tiempo, independientemente del estado inicial en el que se encuentre.

El problema de aprender una política de control para maximizar la recompensa acumulativa es muy general y abarca muchas áreas más allá del aprendizaje de tareas en robótica. En términos generales, se trata de aprender a controlar procesos secuenciales. 

Esto puede incluir problemas de optimización en la fabricación, donde se debe elegir una secuencia de acciones de producción y la recompensa a maximizar es el valor de los bienes producidos menos los costos asociados. 

También abarca problemas de programación secuencial, como decidir qué taxis enviar para recoger pasajeros en una gran ciudad, donde la recompensa a maximizar es una función del tiempo de espera de los pasajeros y el costo total de combustible para la flota de taxis. 

En general, el interés radica en cualquier tipo de agente que deba aprender a elegir acciones que cambien el estado de su entorno, utilizando una función de recompensa acumulativa para evaluar la calidad de cualquier secuencia de acciones dada.

El problema de aprender una política de control para elegir acciones tiene similitudes con los problemas de aproximación de función discutidos en otros capítulos. Sin embargo, el aprendizaje por refuerzo se diferencia en varios aspectos clave:

1. **Recompensa Demorada**: A diferencia de otros métodos donde cada par de entrenamiento (estado, acción) se presenta de manera clara, en el aprendizaje por refuerzo el entrenador ofrece secuencias de recompensas inmediatas. Esto lleva al problema de la asignación de crédito temporal, donde el agente tiene que determinar qué acciones son responsables de las recompensas obtenidas.

2. **Exploración**: En aprendizaje por refuerzo, la secuencia de acciones elegidas por el agente influye en los ejemplos de entrenamiento. Esto lleva a un dilema entre explorar nuevos estados y acciones para obtener más información, o explotar estados y acciones ya conocidos para maximizar la recompensa acumulativa.

3. **Estados Parcialmente Observables**: A menudo, los sensores del agente no pueden percibir todo el estado del entorno. En estos casos, el agente podría necesitar considerar observaciones previas junto con datos actuales de sensores para tomar decisiones. Además, la mejor política podría implicar acciones diseñadas para mejorar la observabilidad del entorno.

4. **Aprendizaje de por Vida**: A diferencia de tareas aisladas de aproximación de funciones, en aprendizaje de robótica, el agente a menudo tiene que aprender múltiples tareas relacionadas en el mismo entorno. Esto abre la posibilidad de utilizar experiencias o conocimientos previos para reducir la complejidad de muestra al aprender nuevas tareas.

## La tarea de aprender.

The text you've provided gives an excellent overview of the complexities and challenges inherent in reinforcement learning (RL) as it relates to sequential decision-making problems. Here are some key takeaways from the text:

### Challenges in Reinforcement Learning
1. **Delayed Reward**: Unlike supervised learning where each sample has an immediate target value, RL often has a sequence of actions that contribute to the final reward. This introduces the need for temporal credit assignment — figuring out which actions were responsible for the final outcome.
  
2. **Exploration vs. Exploitation**: The agent must balance between exploring new actions to find out their outcomes and exploiting known actions to get the highest immediate reward.

3. **Partially Observable States**: Often, the agent does not have full information about the environment state. It may have to infer hidden variables or remember past states to make a good decision.

4. **Life-long Learning**: The agent may have to adapt its policy over time to perform well in multiple related tasks, which introduces a need for transferring knowledge across tasks.

### Markov Decision Process (MDP) Formulation
- **State Space (S)**: A set of all possible states.
- **Action Space (A)**: A set of all possible actions.
- **Reward Function (r)**: Function that gives a reward based on the state and action.
- **State Transition Function (δ)**: Describes how states transition based on actions.
- **Policy (π)**: A function mapping from states to actions, specifying the agent's behavior.
  
### Cumulative Value and Objective
- **Cumulative Value Function (V)**: Defined as the expected sum of discounted future rewards.
- **Discount Factor (γ)**: Determines how future rewards are discounted relative to immediate ones.
- **Optimal Policy (π\*)**: A policy that maximizes the expected cumulative reward over all states.

The agent's goal is to learn an optimal policy that maximizes the cumulative reward \(V^\pi(s)\) for all states \(s\).

### Example: Simple Grid World
- The text describes a simple grid-world to illustrate these concepts. In this grid world, all moves offer zero immediate rewards except for moves that lead to the goal state \(G\), which offers an immediate reward of 100.
  
In this example, the optimal policy guides the agent along the shortest path to the goal state \(G\) to maximize the discounted cumulative reward.

This well-structured explanation provides valuable insights into how RL works and what challenges it poses compared to more conventional machine learning problems.

## Q-Learning

El texto aborda el problema de cómo un agente puede aprender una política óptima en un entorno arbitrario de aprendizaje por refuerzo (RL). A diferencia del aprendizaje supervisado, el agente en RL no tiene acceso a ejemplos de entrenamiento en la forma \((s, a)\) (estado, acción) con una recompensa inmediata. En su lugar, el agente solo recibe una secuencia de recompensas inmediatas \(r(s_i, a_i)\) para diversas combinaciones de estados \(s\) y acciones \(a\).

### Formas de Aproximar la Política Óptima
1. **Función de Valor \(V^*\)**: Una opción para que el agente aprenda la política óptima es mediante el aprendizaje de la función de valor acumulativo \(V^*\), que representa la suma esperada de recompensas futuras descontadas. El agente puede usar \(V^*\) para comparar diferentes estados y acciones.

2. **Función de Acción-Valor \(Q\)**: En situaciones donde el agente no tiene un conocimiento perfecto de las funciones de recompensa \(r\) y de transición de estado \(\delta\), aprender \(V^*\) no es útil. En tales casos, se puede usar otra función de evaluación, llamada función \(Q\), para aprender la política óptima.

### Limitaciones
- **Conocimiento Imperfecto**: Aprender \(V^*\) es útil solo cuando el agente tiene un conocimiento perfecto de \(r\) y \(\delta\).
  
- **Practicidad**: En problemas del mundo real, como el control de robots, no es posible conocer con certeza todas las funciones de transición y recompensa.

### Q-Function

The \(Q\) function in reinforcement learning, which can be understood as a way to evaluate both states \(s\) and actions \(a\). Specifically, \(Q(s, a)\) represents the maximum expected cumulative reward that an agent can obtain starting from state \(s\) and taking action \(a\) as the first step, followed by an optimal policy thereafter.

### Mathematical Definition
The \(Q\) function is mathematically defined as:

\[
Q(s, a) = r(s, a) + \gamma V^*(\delta(s, a))
\]

Here, \(r(s, a)\) is the immediate reward after taking action \(a\) in state \(s\), and \(V^*(\delta(s, a))\) is the value of the state resulting from taking action \(a\) in state \(s\), discounted by the factor \(\gamma\).

### Advantages of Using \(Q\)-Function
- **Direct Policy Extraction**: Once the \(Q\) function is known, selecting the optimal action \(a\) in state \(s\) becomes straightforward: simply choose the action that maximizes \(Q(s, a)\). Mathematically, this is represented as \(n^*(s) = \arg\max_a Q(s, a)\).
  
- **No Need for Model Information**: Importantly, if the \(Q\) function is learned accurately, the agent doesn't need to know the underlying reward and transition functions (\(r\) and \(\delta\)) of the environment to act optimally.

- **Local Decisions for Global Optimum**: The \(Q\) function captures all necessary information for making a decision in a single number. This eliminates the need for the agent to perform lookahead searches to determine the outcome of each action. 

### Practical Implications
The \(Q\) function provides a way to summarize the "goodness" of taking a specific action in a given state, factoring in both immediate and future rewards. Consequently, learning this function allows the agent to make optimal decisions in complex environments, even when the environment's dynamics are not fully known. 

So, the essence of the \(Q\)-learning algorithm lies in learning this \(Q\) function effectively, which in turn helps the agent to perform optimally in a given environment.

The text describes an algorithm for learning the \(Q\) function, which is crucial for making optimal decisions in reinforcement learning. The algorithm leverages the relationship between \(Q\) and \(V^*\), using iterative approximation to gradually refine the estimates of \(Q\).

### How \(Q\) Learning Works

1. **Initialization**: The algorithm initializes a table to represent the \(Q\) function. Each entry in the table corresponds to a state-action pair \((s, a)\) and contains the learner's current estimate of the actual \(Q\) value for that pair. Initial values can be set to zero or random numbers.

2. **Action Selection and Execution**: The agent observes its current state \(s\), selects an action \(a\), executes it, and then observes the immediate reward \(r\) and the new state \(s'\).

3. **Update Rule**: The \(Q\) value for the state-action pair \((s, a)\) is updated using the formula:
\[
Q(s, a) \leftarrow r + \gamma \max_{a'} Q(s', a')
\]
Here, \(r\) is the immediate reward, \(\gamma\) is the discount factor, and \(s'\) is the new state resulting from the action \(a\). The \(\max_{a'} Q(s', a')\) term represents the estimated maximum future reward when starting from the new state \(s'\).

### Features of \(Q\) Learning Algorithm

- **Model-Free**: The agent doesn't need to know the transition probabilities \(\delta(s, a)\) or the reward function \(r(s, a)\) of the environment. It learns by interacting with the environment.
  
- **Iterative**: The algorithm iteratively updates the \(Q\) values, allowing the agent to refine its policy over time.

- **Convergence**: Under certain conditions (like visiting every state-action pair infinitely often and bounded rewards), the \(Q\) values will converge to the true \(Q\) function, enabling the agent to act optimally.

You've provided a comprehensive overview of several key concepts in Q-Learning, a form of reinforcement learning used to find optimal action-selection policies. Let me break down some of the critical points and elaborate on them:

### Q-Learning Algorithm
The core of Q-Learning involves learning the Q-function, which effectively represents the "quality" or utility of taking a certain action in a particular state, aiming to maximize the expected reward. The Q-function is updated iteratively based on the rewards received and the maximum estimated future rewards, usually described as:

\[
Q(s, a) \leftarrow (1 - \alpha) Q(s, a) + \alpha \left( r + \gamma \max_{a'} Q(s', a') \right)
\]

- \(s\) and \(s'\) are the current and next states, respectively.
- \(a\) is the action taken.
- \(r\) is the immediate reward.
- \(\alpha\) is the learning rate.
- \(\gamma\) is the discount factor for future rewards.

### Exploration vs. Exploitation
The algorithm doesn't specify how actions are chosen, which opens the door for experimentation. You can use a deterministic approach where the action with the highest current Q-value is chosen (exploitation), or a stochastic approach where even actions with lower Q-values have a chance to be picked (exploration). Strategies like \(\epsilon\)-greedy, softmax, or decaying \(\epsilon\) offer a balance between exploration and exploitation.

### Update Sequences and Convergence
Q-Learning doesn't require perfect action sequences for convergence. As long as each state-action pair is encountered infinitely often, the algorithm is guaranteed to converge.

### Efficient Training Strategies
1. **Reverse Chronological Updates**: Instead of updating the Q-values in the order experienced, you can update them in reverse order once an episode is complete, effectively propagating the reward information from the goal state backward.

2. **Experience Replay**: Storing past state-action-reward-next_state tuples and reusing them for training can make the algorithm more data-efficient, especially in environments where gathering new experiences is costly.

3. **Prioritized Sweeping**: This focuses on updating states that have a higher potential to improve the policy.

### Knowing the Environment Dynamics
If the agent knows the state-transition and reward functions, then it can simulate the environment, thereby training more efficiently. This simulation-based approach is closer to model-based reinforcement learning.

Your summary covers both the algorithmic basis and practical considerations of Q-Learning, making it a good foundation for anyone interested in reinforcement learning. Would you like to dive deeper into any of these aspects?
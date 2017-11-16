# Scenario descriptions
## General information
### Console variables (CVARs)
Console variables are variables that can be altered in the Doom console or during startup in the command line. To specify values of CVARs for a specific scenario, enter the phrase `+set <CVAR_name> <value>` (or shorthand `+<CVAR_name> <value>`) in the command line when running the vizdoom executable. For ViZDoom, these phrases can be entered via the `game_args` key in configuration files. For example, to set the value of `num_medikits` to 20, the configuration file would need to contain the following:
```conf
game_args = +set num_medikits 20  # or +num_medikits 20
```
Note that using `=` clears any preexisting game arguments, while using `+=` does not. Additionally, adding negative CVARs requires a special modification.

---
## Open field
### Description
The scenario consists of an open, square arena with several objects scattered throughout as visual cues.

### Maps
**MAP00**: The base map from which all scenarios are made. This is not intended to be used as a training tool.

**MAP01**: The base map from which MAP01x are made. This is not intended to be used as a training tool. The map contains beige-toned floors and walls.

**MAP01A**: Extends MAP01. The agent navigates an open field that may contain visual cues, medikits, poison, and/or demons. The behavior and location of these items, as well as reinforcement learning (RL) rewards and goals, may be modified via the console variables (CVARs) described below. All CVARs are supported.

**MAP01B**: Extends MAP01. The agent navigates through the open field and is teleported back to the center if it gets too close to the walls. This map is meant to supplement position encoding. Only the `spawn_objects` CVAR is available.

**MAP02**: The base map from which MAP02x are made. This is not intended to be used as a training tool. The map contains different gray textures for both the floors and walls in each of the four quadrants.

**MAP02A/B**: Extend MAP02. Analogous to MAP01A/B.

**MAP02C**: The agent and a green column are randomly spawned. The agent must navigate to the green column to receive a reward, at which point the game ends. No CVARs are currently supported.

**MAP03**: The base map from which MAP02x are made. This is not intended to be used as a training tool. The map contains differently colored floors and walls in each of the four quadrants.

**MAP03A/B**: Extend MAP03. Analogous to MAP01A/B.

**MAP04**: The base map from which MAP04x are made. This is not intended to be used as a training tool. The map contains vibrantly textured walls with plain floors.

**MAP04A/B**: Extend MAP04. Analogous to MAP01A/B.

### Console Variables
**num_medikits** (int): number of medikits to spawn randomly

**medikit_health** (float): change in health from picking up medikit

**medikit_reward** (float): RL reward for picking up medikit

**x_min_medikit**, **x_max_medikit**, **y_min_medikit**, **y_max_medikit** (float): bounds for spawning medikit (must be in [-256.0, 256.0])

**num_poison** (int): number of poison to spawn randomly

**poison_health** (float): change in health from picking up poison

**poison_reward** (float): RL reward for picking up poison

**x_min_poison**, **x_max_poison**, **y_min_poison**, **y_max_poison** (float): bounds for spawning poison (must be in [-256.0, 256.0])

**num_demons** (int): number of demons to spawn randomly

**demon_reward** (float): RL reward for killing demon

**x_min_demon**, **x_max_demon**, **y_min_demon**, **y_max_demon** (float): bounds for spawning demon (must be in [-256.0, 256.0])

**spawn_objects** (bool): whether to spawn visual cue objects (green column, barrel, brown tree)

**sector_damage** (int): amount of damage inflicted by floor every 32 tics (halved, rounded down, at skill level 1)

---
## Linear track
### Description
The maze consists of a rectangle with visual cues placed intermittently along the walls. In the classic scenario, Reward Buttons are placed on one (simple) or both (alternating) ends, and the agent must navigate to and activate a Reward Button by pressing `USE` within its vicinity in order to receive a reward.

### Maps
MAP00: The base map from which all scenarios are made. This is not intended to be used as a training tool.

MAP01: A simple linear track. The Reward Button is located on the end opposite of the agent initial position. The episode ends once the agent activates the Reward Button.

MAP01A: Extends MAP01. Instead of the Reward Button being located at the opposite end without variation, it is randomly spawned on the part of the track that is 25% closest to the agent's initial starting point.

MAP01B: Similar to MAP01A, but with spawning in nearest 50% of track (i.e. more difficult than MAP01A).

MAP01C: Similar to MAP01A, but with spawning in nearest 70% of track.

MAP01D: Similar to MAP01A, but with spawning in all of track.

MAP01E: Extends MAP01. The Reward Button is fixed on the opposite end as before, but rather than a binary reward structure of {living reward, end reward}, the agent receives a living reward boost proportional to the distance from the Reward Button, given that the agent is in the sector representing the nearest third of the track to the Reward. In other words, if the agent is in the closest third of the track to the Reward Button, for every time step, its living reward is increased (made less negative).

MAP02: An alternating linear track. Reward Buttons are located on opposite ends of the track, but only one can be activated at any given time. The agent must navigate to activatable reward and press `USE` in its vicinity to receive a reward (as before). The agent must then navigate to the other end and activate that Reward Button. The back-and-forth task continues indefinitely.

MAP03, MAP03A-E: Analogous to MAP01 series, except the Reward Button is flagged with `BUMPSPECIAL` instead of `USESPECIAL`. The agent need only navigate to the vicinity of the Reward Button to receive the reward, rather than activating it with `USE`. Thus the `USE` button is not necessary for performance.

MAP04: Analogous to MAP02, except the Reward Button is flagged with `BUMPSPECIAL` instead of `USESPECIAL` (see above).

### Rewards
Activate Reward Button: +1.0

Living reward: -0.01* (suggested)

*In MAP01E, the living reward (reward per time step, or game tic) is decreased as the agent approaches the Reward Button; specifically, it is increased by a number [0, 0.01]. Thus if the living reward is set to -0.01, the agent can never receive a positive reward for not activating the Reward Button, which, if it occurred, could lead to unintended behavior. To visualize the reward step-by-step yourself, use the `spectator.py` code found in the ViZDoom repo.

---
## W-maze (out of date)
### Description
The maze consists of a single long hallway with three branches, each of which contains a spawn site for a medikit at its end. The length of the branch hallways varies from short (wmaze-short) to long (wmaze-long). The floor causes 10% damage continously. The agent must navigate to the ends of the branches in order to pick up medikits in alternating order: left, middle, right, middle, left, middle... . A red pillar and tree stump are placed on opposite ends of the hallway for visual cues.

### Rewards
Pick up medikit: +100

---

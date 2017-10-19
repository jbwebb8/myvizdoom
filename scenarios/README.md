# Scenario descriptions
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
## Open field
### Description
The scenario consists of an open, square arena with several objects scattered throughout as visual cues.

### Maps
MAP00: The base map from which all scenarios are made. This is not intended to be used as a training tool.

MAP01: With the floor intermittently causing 5% damage, the agent must stay alive as long as possible by picking up medikits, which spawn randomly throughout the open field. A maximum of 5 medikits may be present at any given time.

### Rewards
Pick up medikit: +1.0

Living reward: +0.01 (suggested)

Death penalty: -1.0 (suggested)

---
## W-maze (out of date)
### Description
The maze consists of a single long hallway with three branches, each of which contains a spawn site for a medikit at its end. The length of the branch hallways varies from short (wmaze-short) to long (wmaze-long). The floor causes 10% damage continously. The agent must navigate to the ends of the branches in order to pick up medikits in alternating order: left, middle, right, middle, left, middle... . A red pillar and tree stump are placed on opposite ends of the hallway for visual cues.

### Rewards
Pick up medikit: +100

---

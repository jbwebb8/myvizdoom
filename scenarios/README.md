# Scenario Descriptions
## Linear Track
### Description
The maze consists of a simple rectangle with spawn sites for medikits on either end. The floor causes 10% damage continuously. The agent must alternatively navigate to either end in order to pick up medikits.

### Rewards
Pick up medikit: +100


## W-maze
### Description
The maze consists of a single long hallway with three branches, each of which contains a spawn site for a medikit at its end. The length of the branch hallways varies from short (wmaze-short) to long (wmaze-long). The floor causes 10% damage continously. The agent must navigate to the ends of the branches in order to pick up medikits in alternating order: left, middle, right, middle, left, middle... . A red pillar and tree stump are placed on opposite ends of the hallway for visual cues.

### Rewards
Pick up medikit: +100
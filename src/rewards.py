
# These are stats returned from the minerl environment
stat_names = [
    "damage_dealt",
    "damage_taken",
    "damage_blocked_by_shield"
]


class RewardsCalculator:
    def __init__(self, **kwargs):
        """
        Takes arguments of the form "stat_name": weight

        Ex: RewardsCalculator(damage_dealt=1, damage_taken=20)
        """
        self.stats = {}

        for k, v in kwargs.items():
            assert k in stat_names, f"Invalid stat: {k}"

            # Each stat entry corresponds to a tuple of (weight, current)
            # Edit: it's actually a list because tuples are immutable
            # BUT TREAT IT LIKE A TUPLE!!!!
            self.stats[k] = [v, 0]

    def get_rewards(self, obs):

        rewards = []

        for k, v in self.stats.items():
            curr = int(obs[k][k])

            if curr > v[1]:
                # Reward is the amount of change to each stat
                rewards.append((curr - v[1]) * v[0])

                # Update the current value
                v[1] = curr

        return sum(rewards)


if __name__ == "__main__":
    # unit tests are cringe
    rc = RewardsCalculator(
        damage_dealt=1,
        damage_taken=20,
        # poop=4
    )

    obs1 = {
        "damage_dealt": {
            "damage_dealt": 0
        },
        "damage_taken": {
            "damage_taken": 0
        }
    }

    print(rc.get_rewards(obs1))

    obs2 = {
        "damage_dealt": {
            "damage_dealt": 10
        },
        "damage_taken": {
            "damage_taken": 2
        }
    }

    print(rc.get_rewards(obs2))

import time
import random

class Agent:
    def __init__(self, name):
        self.name = name
        self.health = 100
        self.position = (0, 0)
        self.last_actions = []  # Queue to store past actions
        self.can_chrono_sync = True
        self.chrono_sync_cooldown = 45  # seconds
        self.last_sync_time = 0

    def log_action(self, action_description):
        """Logs an action to the history, maintaining only the last 5 seconds worth."""
        current_time = time.time()
        # Add the action with a timestamp
        self.last_actions.append((current_time, action_description))
        # Prune actions older than 5 seconds (the max lookback time)
        self.last_actions = [act for act in self.last_actions if current_time - act[0] <= 5]

    def move(self, direction):
        action_desc = f"Move {direction}"
        print(f"{self.name} is performing action: {action_desc}")
        # ... actual movement logic would go here ...
        self.log_action(action_desc)

    def fire_weapon(self):
        action_desc = "Fire Weapon"
        print(f"{self.name} is performing action: {action_desc}")
        # ... actual firing logic would go here ...
        self.log_action(action_desc)

    def calculate_probable_future(self):
        """A placeholder function to simulate calculating a probable future."""
        # In a real implementation, this would analyze current trajectories,
        # enemy positions, and physics to predict the immediate future.
        print(f"{self.name} is calculating probable future...")
        # For this example, we just return a simple prediction
        possible_futures = ["incoming_attack", "clear"]
        # Let's assume a 70% chance of an attack coming
        return random.choices(possible_futures, weights=[0.7, 0.3], k=1)[0]

    def activate_chrono_sync(self, target, sync_type, sync_point="past"):
        """
        Activates the Chrono-Sync Protocol.

        Args:
            target (Agent): The agent to sync with (self or an ally).
            sync_type (str): The type of sync - 'action', 'dodge', or 'resolve'.
            sync_point (str): 'past' or 'future'.
        """
        current_time = time.time()

        # Check if the ability is on cooldown
        if not self.can_chrono_sync:
            time_since_last_sync = current_time - self.last_sync_time
            if time_since_last_sync < self.chrono_sync_cooldown:
                print(f"Chrono-Sync Protocol is on cooldown. {int(self.chrono_sync_cooldown - time_since_last_sync)} seconds remaining.")
                return False
            else:
                # Cooldown has passed, reset the ability
                self.can_chrono_sync = True

        if not self.can_chrono_sync:
            return False

        print(f"\n{self.name} activates Chrono-Sync Protocol on {target.name} ({sync_point}-{sync_type})!")

        # Handle the different sync types
        if sync_point == "past" and sync_type == "action":
            success = self._echoed_action(target)

        elif sync_point == "future" and sync_type == "dodge":
            success = self._predictive_dodge(target)

        elif sync_type == "resolve":
            success = self._shared_resolve(target)
        else:
            print("Invalid sync parameters.")
            success = False

        # If the sync was successful (or attempted, for future), activate cooldown
        if success is not False: # Specifically check for False, as None is a possible success state for future
            self.can_chrono_sync = False
            self.last_sync_time = current_time
            print(f"Chrono-Sync Protocol cooldown started.")
        return success

    def _echoed_action(self, target):
        """Handles the Echoed Action effect."""
        if not target.last_actions:
            print("Failed: No recent actions to echo.")
            return False

        # Get the most recent past action
        timestamp, action_to_repeat = target.last_actions[-1]
        print(f"Echoing past action: '{action_to_repeat}'")
        # Execute the action without cost (simulated here by just printing)
        # In a real game, you would call the relevant method without consuming resources.
        print(f"** ECHO ** {target.name} re-performs: {action_to_repeat} (No energy cost)")
        return True

    def _predictive_dodge(self, target):
        """Handles the Predictive Dodge effect."""
        future_prediction = self.calculate_probable_future()

        # Check for the 15% failure chance
        if random.random() < 0.15:
            print("Chrono-Sync failed! Future timeline was too unstable.")
            return False

        if future_prediction == "incoming_attack":
            print("Prediction: Incoming attack detected! Executing Predictive Dodge.")
            print(f"** DODGE ** {target.name} seamlessly sidesteps an unseen threat.")
        else:
            print("Prediction: No immediate threats detected. No dodge necessary.")
        # We return None/True even if no dodge was needed, as the ability was successfully used.
        return True

    def _shared_resolve(self, target):
        """Handles the Shared Resolve effect."""
        # Check if the target is an ally (not self)
        if target.name == self.name:
            print("Failed: Shared Resolve must be used on an ally.")
            return False

        # Look for a successful past action in the ally's history
        # For this example, we'll just assume any action was a success.
        if target.last_actions:
            _, past_success = target.last_actions[-1]
            print(f"Channeling ally's past success: '{past_success}'")
            print(f"** RESOLVE ** {self.name} and {target.name} gain enhanced accuracy on their next attack!")
            return True
        else:
            print("Failed: Ally has no recent successful actions to channel.")
            return False

# --- Demo Usage ---
if __name__ == "__main__":
    print("Creating agents...")
    agent_007 = Agent("Agent-007")
    ally_agent = Agent("Agent-008")

    print("\n--- Simulating Actions ---")
    agent_007.move("North")
    time.sleep(1)  # Simulate a small delay
    agent_007.fire_weapon()
    time.sleep(1)
    ally_agent.move("East")

    print("\n--- Using Chrono-Sync Protocol ---")
    # Agent uses Echoed Action on themselves
    agent_007.activate_chrono_sync(agent_007, "action", "past")

    # Agent tries Predictive Dodge
    agent_007.activate_chrono_sync(agent_007, "dodge", "future")

    # Agent uses Shared Resolve on an ally
    agent_007.activate_chrono_sync(ally_agent, "resolve")

    print("\n--- Trying to use it again too soon ---")
    agent_007.activate_chrono_sync(agent_007, "action", "past")
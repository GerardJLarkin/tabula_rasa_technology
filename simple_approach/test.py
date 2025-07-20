import random

def random_choice():
    """Simulate some function whose output (0 or 1) we branch on."""
    return random.randint(0, 1)

def process_branch(max_iterations, iteration=1):
    """
    Recursive “loop” that:
     - stops after max_iterations
     - calls random_choice()
     - branches on its output
     - on one branch, calls random_choice() again for a nested branch
    """
    if iteration > max_iterations:
        print("✅ Reached max iterations. Stopping.")
        return

    val = random_choice()
    print(f"Iteration {iteration}: random_choice() → {val}")

    if val == 1:
        # Branch 1
        print(f"  🔀 Branch 1: doing the '1' path")
    else:
        # Branch 0 → nested function call + nested branch
        print(f"  🔽 Branch 0: calling random_choice() again")
        nested_val = random_choice()
        print(f"    Nested call → {nested_val}")
        if nested_val == 1:
            print("    ↳ Nested Branch 1: doing the '1' path")
        else:
            print("    ↳ Nested Branch 0: doing the '0' path")

    # Recurse to the next iteration
    process_branch(max_iterations, iteration + 1)

# Kick off the recursive “loop” for 5 iterations:
process_branch(5)

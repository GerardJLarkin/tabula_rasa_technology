from typing import Callable, List
import numpy as np

class ArrayGroup:
    """Holds one reference array + its member arrays, updating the reference via reference_fn."""
    def __init__(
        self,
        reference_fn: Callable[[List[np.ndarray]], np.ndarray],
        initial_members: List[np.ndarray] = None
    ):
        self.reference_fn = reference_fn
        self.members: List[np.ndarray] = []
        self.reference: np.ndarray = None
        if initial_members:
            for a in initial_members:
                self.add_member(a)

    def add_member(self, arr: np.ndarray):
        arr = np.array(arr, copy=False)
        self.members.append(arr)
        # recompute the reference archetype over all members
        self.reference = self.reference_fn(self.members)


class ArrayGroupManager:
    """
    Maintains multiple ArrayGroup instances.
    On add_array:
      - compares against each group's reference
      - if best similarity ≥ threshold: adds to that group (and updates its reference)
      - otherwise makes a new group whose reference starts as the new array
    """
    def __init__(
        self,
        compare_fn: Callable[[np.ndarray, np.ndarray], float],
        reference_fn: Callable[[List[np.ndarray]], np.ndarray],
        threshold: float
    ):
        self.compare_fn = compare_fn
        self.reference_fn = reference_fn
        self.threshold = threshold
        self.groups: List[ArrayGroup] = []

    def add_array(self, arr: np.ndarray):
        arr = np.array(arr, copy=False)
        # find the group with highest similarity to its reference
        best_group = None
        best_sim = -np.inf
        for group in self.groups:
            id1, id2, sim = self.compare_fn(group.reference, arr)
            if sim < best_sim:
                best_sim, best_group = sim, group

        # if no group or best_sim below threshold → new group
        if best_group is None or best_sim < self.threshold:
            new_group = ArrayGroup(self.reference_fn, initial_members=[arr])
            self.groups.append(new_group)
        else:
            best_group.add_member(arr)

    def get_all_groups(self) -> List[List[np.ndarray]]:
        """
        Returns a list of groups, each represented as
        [reference_array, member1, member2, ...]
        """
        return [
            [g.reference] + g.members
            for g in self.groups
        ]


# # -- your bespoke functions --
# def my_compare(a: np.ndarray, b: np.ndarray) -> float:
#     # e.g. cosine similarity, correlation, whatever you choose
#     return np.dot(a, b) / (np.linalg.norm(a)*np.linalg.norm(b))

# def my_reference(members: List[np.ndarray]) -> np.ndarray:
#     # e.g. your custom “centroid” or medoid logic
#     stacked = np.stack(members, axis=0)
#     return stacked.mean(axis=0)

# # -- instantiate the manager --
# mgr = ArrayGroupManager(
#     compare_fn=my_compare,
#     reference_fn=my_reference,
#     threshold=0.25
# )

# # -- stream in new data --
# for new_arr in stream_of_new_arrays:
#     mgr.add_array(new_arr)

# # -- inspect groups --
# all_groups = mgr.get_all_groups()
# for idx, grp in enumerate(all_groups):
#     ref = grp[0]
#     members = grp[1:]
#     print(f"Group {idx}: {len(members)} members; reference shape = {ref.shape}")
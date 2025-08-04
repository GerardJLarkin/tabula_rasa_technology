from typing import Callable
import numpy as np

class ArrayGroup:
    """
    Holds one reference array + a count of how many arrays contributed to it.
    On each add, updates reference via inc_ref_fn(current_ref, new_arr, count).
    """
    def __init__(
        self,
        initial_array: np.ndarray,
        inc_ref_fn: Callable[[np.ndarray, np.ndarray, int], np.ndarray]
    ):
        self.reference = np.array(initial_array, copy=True)
        self.inc_ref_fn = inc_ref_fn
        self.count = 1

    def add_member(self, arr: np.ndarray):
        arr = np.array(arr, copy=False)
        # compute new reference from the old reference, new array, and how many we've seen
        self.reference = self.inc_ref_fn(self.reference, arr, self.count)
        self.count += 1

    def as_list(self):
        """Return [reference_array, count]"""
        return [self.reference, self.count]


class ArrayGroupManager:
    """
    Maintains multiple ArrayGroup instances.
    On add_array:
      - compares the new array to each group's reference
      - if best similarity ≥ threshold: adds to that group (and updates its reference & count)
      - otherwise makes a new group with count=1 and reference=new array
    """
    def __init__(
        self,
        compare_fn: Callable[[np.ndarray, np.ndarray], float],
        inc_ref_fn: Callable[[np.ndarray, np.ndarray, int], np.ndarray],
        threshold: float
    ):
        self.compare_fn = compare_fn
        self.inc_ref_fn = inc_ref_fn
        self.threshold = threshold
        self.groups: list[ArrayGroup] = []

    def add_array(self, arr: np.ndarray):
        arr = np.array(arr, copy=False)
        best_group = None
        best_sim = -np.inf

        # find the most-similar existing reference
        for group in self.groups:
            id1, id2, sim = self.compare_fn(group.reference, arr)
            if sim > best_sim:
                best_sim, best_group = sim, group

        # decide whether to join or start new
        if best_group is None or best_sim < self.threshold:
            # new reference group
            new_group = ArrayGroup(arr, self.inc_ref_fn)
            self.groups.append(new_group)
        else:
            best_group.add_member(arr)

    def get_all_groups(self):
        """
        Returns a list of [reference_array, count] for each group.
        """
        return [g.as_list() for g in self.groups]


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
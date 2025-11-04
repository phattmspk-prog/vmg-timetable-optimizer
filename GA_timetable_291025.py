import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
from itertools import combinations
import argparse
import collections # Added for defaultdict in violation calculations
# Define full-scale synthetic data based on the model
# Branches: 10 total, 10 in Biên Hòa
#branches = ['VTS', 'PVT', 'TĐH', 'NKH', 'B5', 'B6', 'B7', 'B8', 'B9', 'B10']
branches = ['VTS', 'PVT', 'TĐH', 'NKH']
branches_bh = branches # Biên Hòa area (modified to include all 10 branches)
# Rooms: 5 per branch, total 50
num_rooms_per_branch = 12
room_ids = list(range(1, len(branches) * num_rooms_per_branch + 1))
branch_of_room = {}
room_ids_per_branch = {}
room_counter = 1
for b in branches:
    rooms_b = list(range(room_counter, room_counter + num_rooms_per_branch))
    room_ids_per_branch[b] = rooms_b
    for r in rooms_b:
        branch_of_room[r] = b
    room_counter += num_rooms_per_branch
# Capacities: random 20-40
cap = {r: random.randint(20, 40) for r in room_ids}
# Classes: 50 total, assigned to branches randomly
num_classes = 250
class_ids = list(range(1, num_classes + 1))
class_branch = {l: random.choice(branches) for l in class_ids}
class_size = {l: random.randint(10, 30) for l in class_ids}
# Modification: Assign levels (A, B, C, D, E) sequentially to classes
levels = ['A', 'B', 'C', 'D', 'E']
class_level = {l: levels[(l - 1) % len(levels)] for l in class_ids}
# Teachers: 20 total, assumed all for Biên Hòa area focus
num_teachers = 86
teacher_ids = [f'T{i}' for i in range(1, num_teachers + 1)]
def teacher_index(e):
    return int(e[1:]) - 1
# NEW: Define teacher types: full-time (fewer, with limits) and part-time (no limits)
num_full_time = 20 # Fewer full-time teachers
teacher_type = {e: 'full-time' if teacher_index(e) < num_full_time else 'part-time' for e in teacher_ids}
# Min/Max sessions (ca): min 6 (9h), max 24 (36h) for full-time; no limits for part-time
min_max = {e: (6, 24) if teacher_type[e] == 'full-time' else (0, 999) for e in teacher_ids} # High max for part-time effectively no limit
# Time slots: 7 days, 6 slots/day
num_days = 7
slots_per_day = 6
num_slots = num_days * slots_per_day
# Active slots function
def is_active(day, slot):
    return True
# Travel times for Biên Hòa (synthetic, in minutes)
travel_time = {}
for b1, b2 in combinations(branches_bh, 2):
    tt = random.randint(5, 25)
    travel_time[(b1, b2)] = tt
    travel_time[(b2, b1)] = tt
for b in branches_bh:
    travel_time[(b, b)] = 0 # No travel same branch
# Forbidden pairs: >15 min
forbidden_pairs = [(b1, b2) for (b1, b2), tt in travel_time.items() if tt > 15 and b1 != b2]
# Qualifications and availabilities (synthetic random)
np.random.seed(42)
q = np.random.rand(num_teachers, num_classes) > 0.5 # q[e_idx, l-1]
a = np.random.rand(num_teachers, num_slots) > 0.5 # a[e_idx, t-1]
# GA parameters (increased for scale)
population_size = 100
generations = 200
num_sessions = 2 # Per class
gene_length = num_classes * num_sessions * 3 # t, r, e per session
parser = argparse.ArgumentParser(description='Genetic Algorithm for Scheduling')
parser.add_argument('--crossover_rate', type=float, default=0.8, help='Crossover probability')
parser.add_argument('--mutation_rate', type=float, default=0.05, help='Mutation probability')
args = parser.parse_args()
crossover_rate = args.crossover_rate
mutation_rate = args.mutation_rate
def get_day_slot(t):
    t -= 1
    day = t // slots_per_day + 1
    slot = t % slots_per_day + 1
    return day, slot
def are_consecutive(t1, t2):
    d1, s1 = get_day_slot(t1)
    d2, s2 = get_day_slot(t2)
    return d1 == d2 and abs(s2 - s1) == 1
# NEW FUNCTION: Check if session 2 comes after session 1 with proper gap
def check_session_order(t1, t2):
    """
    Validates that session 2 occurs after session 1 with at least 1 day gap.
    Returns True if valid, False otherwise.
    t1: time slot for session 1
    t2: time slot for session 2
    """
    d1, s1 = get_day_slot(t1)
    d2, s2 = get_day_slot(t2)
  
    # Session 2 must be on a later day (at least 2 days apart as per original constraint)
    if d2 <= d1:
        return False
  
    # Must be at least 2 days apart (day difference >= 2)
    if abs(d2 - d1) < 2:
        return False
  
    return True
def create_individual():
    ind = []
    for l in class_ids:
        b = class_branch[l]
        possible_r = [rr for rr in room_ids_per_branch[b] if cap[rr] >= class_size[l]]
        possible_t = [t for t in range(1, num_slots + 1) if is_active(*get_day_slot(t))]
        # Modification: Choose one qualified teacher per class for all sessions
        possible_e = [ee for ee in teacher_ids if q[teacher_index(ee), l-1]]
        # NEW: Prioritize full-time teachers
        qualified_full = [ee for ee in possible_e if teacher_type[ee] == 'full-time']
        qualified_part = [ee for ee in possible_e if teacher_type[ee] == 'part-time']
        # MODIFIED: Ensure one full-time and one part-time if possible
        if qualified_full and qualified_part:
            if random.random() < 0.5:
                e1 = random.choice(qualified_full)
                e2 = random.choice(qualified_part)
            else:
                e1 = random.choice(qualified_part)
                e2 = random.choice(qualified_full)
        elif qualified_full:
            e1 = random.choice(qualified_full)
            e2 = random.choice(qualified_full)
        elif qualified_part:
            e1 = random.choice(qualified_part)
            e2 = random.choice(qualified_part)
        else:
            e1 = random.choice(teacher_ids)
            e2 = random.choice(teacher_ids)
      
        # NEW: Ensure session 1 and session 2 follow the ordering constraint
        # Session 1: can be on any active day except day 7 (to allow day 2+ gap)
        possible_t1 = [t for t in possible_t if get_day_slot(t)[0] <= 5] # Days 1-5 for session 1
        if not possible_t1: # Fallback if no valid slots
            possible_t1 = possible_t
        t1 = random.choice(possible_t1)
        d1, _ = get_day_slot(t1)
      
        # Session 2: must be at least 2 days after session 1
        possible_t2 = [t for t in possible_t if get_day_slot(t)[0] >= d1 + 2]
        if not possible_t2: # Fallback
            possible_t2 = [t for t in possible_t if t != t1]
        t2 = random.choice(possible_t2) if possible_t2 else random.choice(possible_t)
      
        r1 = random.choice(possible_r) if possible_r else random.choice(room_ids_per_branch[b])
        r2 = random.choice(possible_r) if possible_r else random.choice(room_ids_per_branch[b])
      
        ind += [t1, r1, e1, t2, r2, e2]
    return ind
def decode(ind):
    assignments = {}
    idx = 0
    for l in class_ids:
        sessions = []
        for _ in range(num_sessions):
            t, r, e = ind[idx:idx + 3]
            sessions.append((t, r, e))
            idx += 3
        assignments[l] = sessions
    return assignments
def calculate_violations_and_objective(assignments):
    hard_violations = 0
    movements = 0
    # 1. Exactly 2 unique sessions per class, not same t
    for l, sess in assignments.items():
        ts = [s[0] for s in sess]
        if len(set(ts)) < num_sessions:
            hard_violations += 1
  
    # 2. Session separation: at least 1 day apart (abs(day diff) >= 2)
    for l, sess in assignments.items():
        days = sorted([get_day_slot(s[0])[0] for s in sess])
        if len(days) == 2 and abs(days[1] - days[0]) < 2:
            hard_violations += 1
  
    # NEW: 2c. Session ordering constraint - Session 2 must come after Session 1
    for l, sess in assignments.items():
        if len(sess) >= 2:
            t1 = sess[0][0] # Session 1 time
            t2 = sess[1][0] # Session 2 time
            if not check_session_order(t1, t2):
                hard_violations += 1
  
    # 3. Capacity, branch match, active slot
    for l, sess in assignments.items():
        b = class_branch[l]
        for t, r, e in sess:
            day, slot = get_day_slot(t)
            if not is_active(day, slot):
                hard_violations += 1
            if cap[r] < class_size[l]:
                hard_violations += 1
            if branch_of_room[r] != b:
                hard_violations += 1
    # 4. Room exclusivity per slot
    room_usage = {}
    for l, sess in assignments.items():
        for t, r, e in sess:
            key = (t, r)
            room_usage[key] = room_usage.get(key, 0) + 1
    for count in room_usage.values():
        if count > 1:
            hard_violations += (count - 1)
    # 5. Teacher feasibility: no overlap, qualified, available
    teacher_usage = {}
    for l, sess in assignments.items():
        for t, r, e in sess:
            if not q[teacher_index(e), l-1]:
                hard_violations += 1
            if not a[teacher_index(e), t-1]:
                hard_violations += 1
            key = (t, e)
            teacher_usage[key] = teacher_usage.get(key, 0) + 1
    for count in teacher_usage.values():
        if count > 1:
            hard_violations += (count - 1)
    # 6. Teacher load
    teacher_load = {e: 0 for e in teacher_ids}
    for l, sess in assignments.items():
        for t, r, e in sess:
            teacher_load[e] += 1
    for e, load in teacher_load.items():
        mn, mx = min_max[e]
        # NEW: Only apply load violations for full-time teachers
        if teacher_type[e] == 'full-time':
            if load < mn:
                hard_violations += (mn - load)
            if load > mx:
                hard_violations += (load - mx)
    # 7. Travel: count movements in BH for consecutive slots in different branches # Updated comment to reflect modification
    teacher_assign = {e: [] for e in teacher_ids}
    for l, sess in assignments.items():
        for t, r, e in sess:
            b = branch_of_room[r]
            teacher_assign[e].append((t, b))
    for e, tba in teacher_assign.items():
        tba.sort(key=lambda x: x[0])
        for i in range(len(tba) - 1):
            t1, b1 = tba[i]
            t2, b2 = tba[i + 1]
            if are_consecutive(t1, t2):
                if b1 in branches_bh and b2 in branches_bh:
                    # Removed: if (b1, b2) in forbidden_pairs: hard_violations += 1 # Hard violation for >15 min travel removed
                    if b1 != b2:
                        movements += 1
    # NEW: Ensure every teacher is assigned and no two classes have the same teacher
    teacher_classes = collections.defaultdict(set)
    for l, sess in assignments.items():
        for s in sess:
            e = s[2]
            teacher_classes[e].add(l)
    # Penalize sharing (more than one class per teacher)
    for e, clss in teacher_classes.items():
        num_clss = len(clss)
        if num_clss > 1:
            hard_violations += (num_clss - 1)
    # Penalize unused teachers
    used_teachers = len(teacher_classes) # Number of teachers with at least one class
    hard_violations += (num_teachers - used_teachers)
    # MODIFIED: Penalize if sessions of a class do not have one full-time and one part-time teacher
    for l, sess in assignments.items():
        types = set(teacher_type[s[2]] for s in sess)
        if len(types) != 2:
            hard_violations += 1
    return hard_violations, movements
def fitness(ind):
    ass = decode(ind)
    v, m = calculate_violations_and_objective(ass)
    #return 1000 * v + m # MODIFIED: Restored composite fitness to prioritize minimizing violations
    return m
def selection(pop, k=5): # Increased k for better selection in larger space
    selected = []
    for _ in range(len(pop)):
        cand = random.sample(pop, k)
        selected.append(min(cand, key=fitness))
    return selected
def crossover(p1, p2):
    point = random.randint(1, len(p1) - 2)
    c1 = p1[:point] + p2[point:]
    c2 = p2[:point] + p1[point:]
    return c1, c2
def mutate(ind):
    for _ in range(random.randint(1, 3)): # Multiple mutations for larger genome
        if random.random() < mutation_rate:
            pos = random.randint(0, len(ind) - 1)
            session_idx = pos // 3
            l = (session_idx // num_sessions) + 1
            b = class_branch[l]
            possible_r = [rr for rr in room_ids_per_branch[b] if cap[rr] >= class_size[l]]
            possible_t = [t for t in range(1, num_slots + 1) if is_active(*get_day_slot(t))]
          
            if pos % 3 == 0: # t (time slot mutation)
                # NEW: Ensure session ordering when mutating time slots
                class_start_idx = (l - 1) * num_sessions * 3
                session_num = (session_idx % num_sessions) # 0 for session 1, 1 for session 2
              
                if session_num == 0: # Mutating session 1
                    # Session 1 should be on days 1-5 to allow for session 2
                    possible_t1 = [t for t in possible_t if get_day_slot(t)[0] <= 5]
                    if possible_t1:
                        new_t1 = random.choice(possible_t1)
                        ind[pos] = new_t1
                        # Adjust session 2 if needed to maintain ordering
                        t2_pos = class_start_idx + 3 # Position of session 2 time
                        t2 = ind[t2_pos]
                        if not check_session_order(new_t1, t2):
                            d1, _ = get_day_slot(new_t1)
                            possible_t2 = [t for t in possible_t if get_day_slot(t)[0] >= d1 + 2]
                            if possible_t2:
                                ind[t2_pos] = random.choice(possible_t2)
                    else:
                        ind[pos] = random.choice(possible_t)
                      
                else: # Mutating session 2
                    # Session 2 must be at least 2 days after session 1
                    t1_pos = class_start_idx # Position of session 1 time
                    t1 = ind[t1_pos]
                    d1, _ = get_day_slot(t1)
                    possible_t2 = [t for t in possible_t if get_day_slot(t)[0] >= d1 + 2]
                    if possible_t2:
                        ind[pos] = random.choice(possible_t2)
                    else:
                        ind[pos] = random.choice(possible_t)
                      
            elif pos % 3 == 1: # r
                ind[pos] = random.choice(possible_r) if possible_r else ind[pos]
            else: # e
                possible_e = [ee for ee in teacher_ids if q[teacher_index(ee), l-1]]
                # NEW: Prioritize full-time teachers in mutation
                qualified_full = [ee for ee in possible_e if teacher_type[ee] == 'full-time']
                qualified_part = [ee for ee in possible_e if teacher_type[ee] == 'part-time']
                # MODIFIED: Prefer opposite type to the other session's teacher
                class_start_idx = (l - 1) * num_sessions * 3
                session_num = session_idx % num_sessions
                other_e_pos = class_start_idx + 5 if session_num == 0 else class_start_idx + 2
                other_e = ind[other_e_pos]
                other_type = teacher_type[other_e]
                qualified_opposite = qualified_full if other_type == 'part-time' else qualified_part
                if qualified_opposite:
                    new_e = random.choice(qualified_opposite)
                elif qualified_full:
                    new_e = random.choice(qualified_full)
                elif qualified_part:
                    new_e = random.choice(qualified_part)
                else:
                    new_e = random.choice(teacher_ids)
                ind[pos] = new_e
    return ind
# Run GA
random.seed(42)
pop = [create_individual() for _ in range(population_size)]
fitness_history = []
for gen in range(generations):
    pop = selection(pop)
    new_pop = []
    for i in range(0, len(pop), 2):
        p1 = pop[i]
        p2 = pop[i + 1] if i + 1 < len(pop) else pop[0]
        if random.random() < crossover_rate:
            c1, c2 = crossover(p1, p2)
        else:
            c1, c2 = p1[:], p2[:]
        c1 = mutate(c1)
        c2 = mutate(c2)
        new_pop += [c1, c2]
    pop = new_pop[:population_size] # Trim if odd
    best = min(pop, key=fitness)
    best_fit = fitness(best)
    fitness_history.append(best_fit)
    if gen % 50 == 0:
        print(f"Generation {gen}: Best Fitness = {best_fit}")
# Save fitness evolution chart
plt.figure()
plt.plot(fitness_history)
plt.xlabel('Generation')
plt.ylabel('Fitness (1000*violations + movements)') # Updated to reflect composite fitness
plt.title('Fitness Evolution')
plt.savefig('fitness.png')
# Results
best = min(pop, key=fitness)
best_ass = decode(best)
v, m = calculate_violations_and_objective(best_ass)
# Build auxiliary structures for conflict detection and suggestions
import collections # Already present, reiterated for clarity
teacher_usage = collections.defaultdict(int)
room_usage = collections.defaultdict(int)
teacher_levels = collections.defaultdict(set)
for l, sess in best_ass.items():
    for _, _, e in sess: # Modified to add levels for all teachers in the class sessions
        teacher_levels[e].add(class_level[l])
    for t, r, e in sess:
        teacher_usage[(t, e)] += 1
        room_usage[(t, r)] += 1
def detect_teacher_conflicts(assignments, teacher_usage):
    from collections import defaultdict
    teacher_slot_classes = defaultdict(list)
    for l, sess in assignments.items():
        for t, r, e in sess:
            teacher_slot_classes[(e, t)].append(l)
    conflicts = []
    for (e, t), classes in teacher_slot_classes.items():
        if len(classes) > 1:
            conflicts.append((e, t, classes))
    return conflicts
def generate_suggestions(conflict, assignments, teacher_usage, room_usage, teacher_levels):
    e, t, classes = conflict
    day, slot = get_day_slot(t)
    print(f"Suggestions for resolving conflict - Teacher {e} at Day {day}, Slot {slot} with Classes {', '.join(map(str, classes))}:")
    for l in classes:
        print(f" For Class {l} (Level: {class_level[l]}):")
      
        # Option 1: Reassign to alternative teacher, prioritizing level compatibility
        possible_e = []
        for ee in teacher_ids:
            if ee == e:
                continue
            idx = teacher_index(ee)
            if q[idx, l-1] and a[idx, t-1] and teacher_usage[(t, ee)] == 0:
                possible_e.append(ee)
        if possible_e:
            # Prioritize teachers already teaching the same level
            possible_e.sort(key=lambda ee: 1 if class_level[l] in teacher_levels[ee] else 0, reverse=True)
            possible_e = possible_e[:3] # Limit to top 3
            print(" Option 1: Reassign to alternative teachers (prioritized by level compatibility): " + ", ".join(possible_e))
        else:
            print(" Option 1: No alternative teachers available.")
      
        # Option 2: Relocate session to available slot for current teacher
        sess = assignments[l]
        other_t = [s[0] for s in sess if s[0] != t][0]
        d, _ = get_day_slot(t)
        other_d, _ = get_day_slot(other_t)
        is_session1 = d < other_d
        possible_t_new = []
        b = class_branch[l]
        possible_r = [rr for rr in room_ids_per_branch[b] if cap[rr] >= class_size[l]]
        for t_new in range(1, num_slots + 1):
            if t_new == t:
                continue
            day_new, slot_new = get_day_slot(t_new)
            if not is_active(day_new, slot_new):
                continue
            if not a[teacher_index(e), t_new - 1]:
                continue
            if teacher_usage[(t_new, e)] > 0:
                continue
            # Check session order
            if is_session1:
                if not check_session_order(t_new, other_t):
                    continue
            else:
                if not check_session_order(other_t, t_new):
                    continue
            # Check for free room
            has_free_room = any(room_usage[(t_new, r)] == 0 for r in possible_r)
            if has_free_room:
                possible_t_new.append(t_new)
        if possible_t_new:
            possible_t_new = possible_t_new[:3] # Limit to top 3
            slot_desc = [f"{tt} (Day {get_day_slot(tt)[0]}, Slot {get_day_slot(tt)[1]})" for tt in possible_t_new]
            print(" Option 2: Relocate session to alternative slots: " + ", ".join(slot_desc))
        else:
            print(" Option 2: No alternative slots available.")
def resolve_conflict_automatically(conflict, assignments, teacher_usage, room_usage, teacher_levels):
    e, t, classes = conflict
    day, slot = get_day_slot(t)
  
    # Keep the first class; resolve the others
    kept_class = classes[0]
    for l in classes[1:]:
        # Identify the conflicting session for this class
        sess = assignments[l]
        conflicting_session_idx = next(i for i, s in enumerate(sess) if s[0] == t)
        other_session_idx = 1 - conflicting_session_idx
        other_t = sess[other_session_idx][0]
        is_session1 = t < other_t # Determine if conflicting is session 1
      
        old_t = t
        old_r = sess[conflicting_session_idx][1]
        old_e = e
      
        b = class_branch[l]
        possible_r = [rr for rr in room_ids_per_branch[b] if cap[rr] >= class_size[l]]
      
        # MODIFIED Option 1: Reassign only the conflicting session to a new teacher of opposite type
        other_e = sess[other_session_idx][2]
        other_type = teacher_type[other_e]
        possible_e_new = []
        qualified_opposite = [ee for ee in teacher_ids if teacher_type[ee] != other_type and q[teacher_index(ee), l-1] and a[teacher_index(ee), old_t - 1] and teacher_usage[(old_t, ee)] == 0]
        qualified_same = [ee for ee in teacher_ids if teacher_type[ee] == other_type and q[teacher_index(ee), l-1] and a[teacher_index(ee), old_t - 1] and teacher_usage[(old_t, ee)] == 0]
        possible_e_new = qualified_opposite or qualified_same
        if possible_e_new:
            # Prioritize by level compatibility
            possible_e_new.sort(key=lambda ee: 1 if class_level[l] in teacher_levels[ee] else 0, reverse=True)
            new_e = possible_e_new[0]
          
            # Apply reassignment to conflicting session only
            teacher_usage[(old_t, old_e)] -= 1
            teacher_usage[(old_t, new_e)] += 1
            sess[conflicting_session_idx] = (old_t, old_r, new_e)
          
            # Update teacher_levels
            teacher_levels[old_e].discard(class_level[l]) if not any(s[2] == old_e for s in sess) else None
            teacher_levels[new_e].add(class_level[l])
          
            print(f"Resolved conflict for Class {l}: Reassigned conflicting session teacher from {old_e} to {new_e}")
            continue
      
        # Option 2: Relocate the conflicting session to an available slot
        possible_t_new = []
        for t_new in range(1, num_slots + 1):
            if t_new == old_t or t_new == other_t:
                continue
            day_new, slot_new = get_day_slot(t_new)
            if not is_active(day_new, slot_new):
                continue
            if not a[teacher_index(old_e), t_new - 1]:
                continue
            if teacher_usage[(t_new, old_e)] > 0:
                continue
            # Check session order
            if is_session1:
                if not check_session_order(t_new, other_t):
                    continue
            else:
                if not check_session_order(other_t, t_new):
                    continue
            # Check for at least one free room
            free_rooms = [rr for rr in possible_r if room_usage[(t_new, rr)] == 0]
            if free_rooms:
                possible_t_new.append((t_new, free_rooms))
      
        if possible_t_new:
            t_new, free_rooms = possible_t_new[0]
            new_r = free_rooms[0] # Pick the first free room
          
            # Apply relocation
            teacher_usage[(old_t, old_e)] -= 1
            teacher_usage[(t_new, old_e)] += 1
            room_usage[(old_t, old_r)] -= 1
            room_usage[(t_new, new_r)] += 1
            sess[conflicting_session_idx] = (t_new, new_r, old_e)
          
            print(f"Resolved conflict for Class {l}: Relocated session from slot {old_t} (Day {day}, Slot {slot}) to slot {t_new} (Day {get_day_slot(t_new)[0]}, Slot {get_day_slot(t_new)[1]}) in Room {new_r}")
            continue
      
        print(f"Could not automatically resolve conflict for Class {l}")
# Detect and handle conflicts with automated resolution
conflicts = detect_teacher_conflicts(best_ass, teacher_usage)
if conflicts:
    print("Detected teacher conflicts:")
    for conflict in conflicts:
        e, t, classes = conflict
        day, slot = get_day_slot(t)
        print(f" Teacher {e} at Day {day}, Slot {slot}: Classes {', '.join(map(str, classes))}")
  
    print("\nAttempting automated resolution...")
    for conflict in conflicts:
        resolve_conflict_automatically(conflict, best_ass, teacher_usage, room_usage, teacher_levels)
  
    # Re-detect conflicts after resolutions
    conflicts = detect_teacher_conflicts(best_ass, teacher_usage)
    if not conflicts:
        print("All teacher conflicts resolved successfully.")
    else:
        print("Some teacher conflicts could not be resolved automatically. Remaining conflicts:")
        for conflict in conflicts:
            e, t, classes = conflict
            day, slot = get_day_slot(t)
            print(f" Teacher {e} at Day {day}, Slot {slot}: Classes {', '.join(map(str, classes))}")
else:
    print("No teacher conflicts detected.")
# NEW: Function to detect room conflicts
def detect_room_conflicts(assignments, room_usage):
    from collections import defaultdict
    room_slot_classes = defaultdict(list)
    for l, sess in assignments.items():
        for t, r, e in sess:
            room_slot_classes[(r, t)].append(l)
    conflicts = []
    for (r, t), classes in room_slot_classes.items():
        if len(classes) > 1:
            conflicts.append((r, t, classes))
    return conflicts
# NEW: Function to automatically resolve room conflicts
def resolve_room_conflict_automatically(conflict, assignments, teacher_usage, room_usage):
    r, t, classes = conflict
    day, slot = get_day_slot(t)
  
    # Keep the first class; resolve the others
    kept_class = classes[0]
    for l in classes[1:]:
        # Identify the conflicting session for this class
        sess = assignments[l]
        conflicting_session_idx = next(i for i, s in enumerate(sess) if s[0] == t)
        other_session_idx = 1 - conflicting_session_idx
        other_t = sess[other_session_idx][0]
        is_session1 = t < other_t # Determine if conflicting is session 1
      
        old_t = t
        old_r = r
        old_e = sess[conflicting_session_idx][2]
      
        b = class_branch[l]
      
        # Strategy 1: Reassign to available room in same branch
        possible_r_new = [rr for rr in room_ids_per_branch[b] if cap[rr] >= class_size[l] and room_usage[(t, rr)] == 0]
        if possible_r_new:
            new_r = possible_r_new[0] # Pick the first available
            room_usage[(old_t, old_r)] -= 1
            room_usage[(old_t, new_r)] += 1
            sess[conflicting_session_idx] = (old_t, new_r, old_e)
            print(f"Resolved room conflict for Class {l}: Reassigned room from {old_r} to {new_r} at slot {old_t} (Day {day}, Slot {slot})")
            continue
      
        # Strategy 2: Adjust time slot if reassignment not possible
        possible_t_new = []
        for t_new in range(1, num_slots + 1):
            if t_new == old_t or t_new == other_t:
                continue
            day_new, slot_new = get_day_slot(t_new)
            if not is_active(day_new, slot_new):
                continue
            if not a[teacher_index(old_e), t_new - 1]:
                continue
            if teacher_usage[(t_new, old_e)] > 0:
                continue
            # Check session order
            if is_session1:
                if not check_session_order(t_new, other_t):
                    continue
            else:
                if not check_session_order(other_t, t_new):
                    continue
            # Check for at least one free room in the branch
            free_rooms = [rr for rr in room_ids_per_branch[b] if cap[rr] >= class_size[l] and room_usage[(t_new, rr)] == 0]
            if free_rooms:
                possible_t_new.append((t_new, free_rooms))
      
        if possible_t_new:
            t_new, free_rooms = possible_t_new[0]
            new_r = free_rooms[0] # Pick the first free room
            # Apply relocation
            teacher_usage[(old_t, old_e)] -= 1
            teacher_usage[(t_new, old_e)] += 1
            room_usage[(old_t, old_r)] -= 1
            room_usage[(t_new, new_r)] += 1
            sess[conflicting_session_idx] = (t_new, new_r, old_e)
            print(f"Resolved room conflict for Class {l}: Relocated session from slot {old_t} (Day {day}, Slot {slot}) in Room {old_r} to slot {t_new} (Day {get_day_slot(t_new)[0]}, Slot {get_day_slot(t_new)[1]}) in Room {new_r}")
            continue
      
        print(f"Could not automatically resolve room conflict for Class {l}")
# NEW: Detect and handle room conflicts after teacher resolution
room_conflicts = detect_room_conflicts(best_ass, room_usage)
if room_conflicts:
    print("Detected room conflicts:")
    for conflict in room_conflicts:
        r, t, classes = conflict
        day, slot = get_day_slot(t)
        print(f" Room {r} at Day {day}, Slot {slot}: Classes {', '.join(map(str, classes))}")
  
    print("\nAttempting automated resolution for room conflicts...")
    for conflict in room_conflicts:
        resolve_room_conflict_automatically(conflict, best_ass, teacher_usage, room_usage)
  
    # Re-detect room conflicts after resolutions
    room_conflicts = detect_room_conflicts(best_ass, room_usage)
    if not room_conflicts:
        print("All room conflicts resolved successfully.")
    else:
        print("Some room conflicts could not be resolved automatically. Remaining conflicts:")
        for conflict in room_conflicts:
            r, t, classes = conflict
            day, slot = get_day_slot(t)
            print(f" Room {r} at Day {day}, Slot {slot}: Classes {', '.join(map(str, classes))}")
    # Recompute violations and movements with updated assignments
    v, m = calculate_violations_and_objective(best_ass)
else:
    print("No room conflicts detected.")
# NEW: Function to detect three or more consecutive slots for teachers on the same day
def detect_consecutive_slot_conflicts(assignments):
    teacher_day_slots = collections.defaultdict(lambda: collections.defaultdict(list))
    teacher_slot_info = {} # (e, t): (l, idx, r)
    for l, sess in assignments.items():
        for idx, (t, r, e) in enumerate(sess):
            day, slot = get_day_slot(t)
            teacher_day_slots[e][day].append(slot)
            teacher_slot_info[(e, t)] = (l, idx, r)
    conflicts = []
    for e in teacher_day_slots:
        for day in teacher_day_slots[e]:
            slots = sorted(set(teacher_day_slots[e][day]))
            seq_start = 0
            for i in range(1, len(slots)):
                if slots[i] != slots[i-1] + 1:
                    seq_len = i - seq_start
                    if seq_len >= 3:
                        third_slot = slots[seq_start + 2]
                        third_t = (day - 1) * slots_per_day + third_slot
                        l, idx, r = teacher_slot_info[(e, third_t)]
                        conflicts.append((e, day, third_t, l, idx, r))
                    seq_start = i
            # Check last sequence
            seq_len = len(slots) - seq_start
            if seq_len >= 3:
                third_slot = slots[seq_start + 2]
                third_t = (day - 1) * slots_per_day + third_slot
                l, idx, r = teacher_slot_info[(e, third_t)]
                conflicts.append((e, day, third_t, l, idx, r))
    return conflicts
# NEW: Function to automatically resolve consecutive slot conflicts
def resolve_consecutive_conflict_automatically(conflict, assignments, teacher_usage, a, q, teacher_levels):
    old_e, day, t, l, idx, r = conflict
    sess = assignments[l]
    old_session = sess[idx]
    assert old_session == (t, r, old_e)
    # Rebuild teacher_day_slots for current state
    teacher_day_slots = collections.defaultdict(lambda: collections.defaultdict(list))
    for ll, ss in assignments.items():
        for tt, rr, ee in ss:
            dday, sslot = get_day_slot(tt)
            teacher_day_slots[ee][dday].append(sslot)
    possible_new_e = []
    slot = get_day_slot(t)[1]
    other_idx = 1 - idx
    other_e = sess[other_idx][2]
    other_type = teacher_type[other_e]
    for ee in teacher_ids:
        if ee == old_e:
            continue
        ee_idx = teacher_index(ee)
        if not q[ee_idx, l - 1]:
            continue
        if not a[ee_idx, t - 1]:
            continue
        if teacher_usage[(t, ee)] > 0:
            continue # Slot occupied by ee
        # MODIFIED: Prefer opposite type
        if teacher_type[ee] != other_type:
            priority = 2  # Higher for opposite
        else:
            priority = 1  # Lower for same
        # Simulate adding the slot and check for new 3+ consecutive
        current_slots = sorted(set(teacher_day_slots[ee][day]))
        new_slots = sorted(set(current_slots + [slot]))
        has_consec = False
        seq_start = 0
        for ii in range(1, len(new_slots)):
            if new_slots[ii] != new_slots[ii-1] + 1:
                if ii - seq_start >= 3:
                    has_consec = True
                    break
                seq_start = ii
        if len(new_slots) - seq_start >= 3:
            has_consec = True
        if has_consec:
            continue
        load = len(current_slots)
        possible_new_e.append((ee, load, priority))
    if not possible_new_e:
        print(f"Could not automatically resolve consecutive conflict for Teacher {old_e}, Class {l}")
        return
    # Sort by priority desc, then load asc
    possible_new_e.sort(key=lambda x: (-x[2], x[1]))
    new_e = possible_new_e[0][0]
    # Apply reassignment
    teacher_usage[(t, old_e)] -= 1
    teacher_usage[(t, new_e)] += 1
    sess[idx] = (t, r, new_e)
    # Update teacher_levels
    still_teaches = any(s[2] == old_e for s in sess)
    if not still_teaches:
        teacher_levels[old_e].discard(class_level[l])
    teacher_levels[new_e].add(class_level[l])
    print(f"Resolved consecutive conflict for Teacher {old_e}, Class {l}: Reassigned to {new_e}")
# NEW: Function to generate suggestions for resolving consecutive slot conflicts
def generate_consecutive_suggestions(conflicts, assignments, a, q):
    if not conflicts:
        return
    print("Detected consecutive slot conflicts:")
    for e, day, third_t, l, idx, r in conflicts:
        _, slot = get_day_slot(third_t)
        print(f" Teacher {e} on Day {day} has 3+ consecutive slots, targeting third slot {slot} for Class {l} Session {idx+1}")
  
    print("\nSuggestions for resolution:")
    # Rebuild teacher_day_slots for current state
    teacher_day_slots = collections.defaultdict(lambda: collections.defaultdict(list))
    for l, sess in assignments.items():
        for t, r, e in sess:
            day, slot = get_day_slot(t)
            teacher_day_slots[e][day].append(slot)
  
    for e, day, third_t, l, idx, r in conflicts:
        possible_new_e = []
        slot = get_day_slot(third_t)[1]
        for ee in teacher_ids:
            if ee == e:
                continue
            ee_idx = teacher_index(ee)
            if not q[ee_idx, l - 1]:
                continue
            if not a[ee_idx, third_t - 1]:
                continue
            if slot in set(teacher_day_slots[ee][day]):
                continue # Slot already occupied by ee
            # Simulate adding the slot and check for new 3+ consecutive
            current_slots = sorted(set(teacher_day_slots[ee][day]))
            new_slots = sorted(set(current_slots + [slot]))
            has_consec = False
            seq_start = 0
            for ii in range(1, len(new_slots)):
                if new_slots[ii] != new_slots[ii-1] + 1:
                    if ii - seq_start >= 3:
                        has_consec = True
                        break
                    seq_start = ii
            if len(new_slots) - seq_start >= 3:
                has_consec = True
            if has_consec:
                continue
            load = len(current_slots)
            possible_new_e.append((ee, load))
        # Sort by load ascending
        possible_new_e.sort(key=lambda x: x[1])
        top = [ee for ee, _ in possible_new_e[:3]]
        _, conflict_slot = get_day_slot(third_t)
        print(f"For conflict of Teacher {e}, Class {l}, Session {idx+1}, Day {day}, Slot {conflict_slot}:")
        if top:
            print(" Reassign to alternative teachers (lowest day load first): " + ", ".join(top))
        else:
            print(" No suitable alternative teachers found.")
# NEW: Detect and handle consecutive slot conflicts after previous resolutions
consec_conflicts = detect_consecutive_slot_conflicts(best_ass)
if consec_conflicts:
    print("Detected consecutive slot conflicts:")
    for e, day, t, l, idx, r in consec_conflicts:
        _, slot = get_day_slot(t)
        print(f" Teacher {e} on Day {day} has 3+ consecutive slots, targeting third slot {slot} for Class {l} Session {idx+1}")
  
    print("\nAttempting automated resolution for consecutive conflicts...")
    for conflict in consec_conflicts:
        resolve_consecutive_conflict_automatically(conflict, best_ass, teacher_usage, a, q, teacher_levels)
  
    # Re-detect consecutive conflicts after resolutions
    consec_conflicts = detect_consecutive_slot_conflicts(best_ass)
    if not consec_conflicts:
        print("All consecutive slot conflicts resolved successfully.")
    else:
        print("Some consecutive slot conflicts could not be resolved automatically. Remaining conflicts:")
        for e, day, t, l, idx, r in consec_conflicts:
            _, slot = get_day_slot(t)
            print(f" Teacher {e} on Day {day} has 3+ consecutive slots, targeting third slot {slot} for Class {l} Session {idx+1}")
  
    # If remaining, provide suggestions
    if consec_conflicts:
        generate_consecutive_suggestions(consec_conflicts, best_ass, a, q)
else:
    print("No consecutive slot conflicts detected.")
# Recompute after all resolutions
v, m = calculate_violations_and_objective(best_ass)
print(f"Best Solution: Violations = {v}, Movements = {m}")
# Schedule DataFrame and export (sample first 10 classes for illustration)
data = []
for l in list(best_ass.keys())[:num_classes]: # Illustrative subset
    for i, (t, r, e) in enumerate(best_ass[l], 1):
        day, slot = get_day_slot(t)
        # Modification: Add 'Level' to the DataFrame
        # NEW: Add 'Teacher Type' to the DataFrame
        data.append({'Class': l, 'Level': class_level[l], 'Session': i, 'Day': day, 'Slot': slot, 'Room': r, 'Branch': class_branch[l], 'Teacher': e, 'Teacher Type': teacher_type[e]})
df = pd.DataFrame(data)
df.to_excel('schedule.xlsx', index=False)
# Illustrative Example Output (first few assignments)
for l in list(best_ass.keys())[:5]:
    # Modification: Include level in the print output
    print(f"Class {l} (Level: {class_level[l]}, Branch: {class_branch[l]}, Size: {class_size[l]}):")
    for i, (t, r, e) in enumerate(best_ass[l], 1):
        day, slot = get_day_slot(t)
        print(f" Session {i}: Day {day}, Slot {slot}, Room {r} (Cap: {cap[r]}, Branch: {branch_of_room[r]}), Teacher {e} (Type: {teacher_type[e]})")
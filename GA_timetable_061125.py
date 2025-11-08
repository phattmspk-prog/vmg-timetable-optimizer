import argparse
import random
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd


# ---------------------------------------------------------------------------
# Data loading utilities
# ---------------------------------------------------------------------------


SLOTS_PER_DAY = 6


def _load_branch_hours(path: str, rooms_df: pd.DataFrame) -> set[tuple[str, int, int]]:
    """
    Return the set of (Branch, Day, Slot) that are Active=1.
    If the 'BranchHours' sheet is missing or empty, assume every (branch,day,slot) is allowed.
    """
    try:
        bh = pd.read_excel(path, sheet_name="BranchHours")
        bh["Branch"] = bh["Branch"].astype(str).str.strip()
        bh["Day"] = pd.to_numeric(bh["Day"], errors="coerce").astype(int)
        bh["Slot"] = pd.to_numeric(bh["Slot"], errors="coerce").astype(int)
        bh["Active"] = pd.to_numeric(bh["Active"], errors="coerce").fillna(0).astype(int)
        allowed = set(map(tuple, bh.loc[bh["Active"] == 1, ["Branch", "Day", "Slot"]]
                          .itertuples(index=False, name=None)))
        if not allowed:
            branches = rooms_df["Branch"].astype(str).str.strip().unique().tolist()
            allowed = {(b, d, s) for b in branches for d in range(1, 8) for s in range(1, 7)}
        return allowed
    except Exception:
        branches = rooms_df["Branch"].astype(str).str.strip().unique().tolist()
        return {(b, d, s) for b in branches for d in range(1, 8) for s in range(1, 7)}


def load_input_data(path: str):
    classes_df = pd.read_excel(path, sheet_name="Classes")
    teachers_df = pd.read_excel(path, sheet_name="Teachers")
    rooms_df = pd.read_excel(path, sheet_name="Rooms")
    availability_df = pd.read_excel(path, sheet_name="Availability")

    class_ids = classes_df["ClassID"].tolist()
    class_branch = {
        row.ClassID: str(row.Branch)
        for row in classes_df.itertuples()
    }
    class_level = {
        row.ClassID: str(row.Level)
        for row in classes_df.itertuples()
    }
    class_size = {
        row.ClassID: int(row.Size)
        for row in classes_df.itertuples()
    }

    teacher_ids = [str(tid) for tid in teachers_df["TeacherID"].tolist()]
    teacher_type = {}
    for row in teachers_df.itertuples():
        tid = str(row.TeacherID)
        type_value = str(row.Type).strip().lower().replace(" ", "-")
        if type_value not in {"full-time", "part-time"}:
            raise ValueError(f"Unsupported teacher type '{row.Type}' for teacher {tid}")
        teacher_type[tid] = type_value

    # Default workload limits – can be adjusted per organisation rules.
    min_max = {}
    for tid in teacher_ids:
        t_type = teacher_type[tid]
        if t_type == "full-time":
            min_max[tid] = (0, 1000)
        else:
            min_max[tid] = (0, 1000)

    qualifications = {}
    for row in teachers_df.itertuples():
        tid = str(row.TeacherID)
        if pd.isna(row.QualifiedLevels):
            qualifications[tid] = set()
            continue
        levels = [level.strip() for level in str(row.QualifiedLevels).split(",") if level.strip()]
        qualifications[tid] = set(levels)

    room_ids = [str(rid) for rid in rooms_df["RoomID"].tolist()]
    branch_of_room = {
        str(row.RoomID): str(row.Branch)
        for row in rooms_df.itertuples()
    }
    room_ids_per_branch: Dict[str, List[str]] = defaultdict(list)
    for room, branch in branch_of_room.items():
        room_ids_per_branch[branch].append(room)

    capacities = {
        str(row.RoomID): int(row.Capacity)
        for row in rooms_df.itertuples()
    }

    days = list(range(1, 8))
    slots_per_day = SLOTS_PER_DAY
    availability: Dict[str, Dict[int, Dict[int, bool]]] = {
        tid: {day: {slot: False for slot in range(1, slots_per_day + 1)} for day in days}
        for tid in teacher_ids
    }
    for row in availability_df.itertuples():
        tid = str(row.TeacherID)
        day = int(row.Day)
        slot = int(row.Slot)
        available = bool(row.Available)
        if tid in availability and day in availability[tid] and slot in availability[tid][day]:
            availability[tid][day][slot] = available

    allowed_branch_slots = _load_branch_hours(path, rooms_df)

    return {
        "class_ids": class_ids,
        "class_branch": class_branch,
        "class_level": class_level,
        "class_size": class_size,
        "teacher_ids": teacher_ids,
        "teacher_type": teacher_type,
        "min_max": min_max,
        "qualifications": qualifications,
        "room_ids": room_ids,
        "branch_of_room": branch_of_room,
        "room_ids_per_branch": room_ids_per_branch,
        "capacities": capacities,
        "availability": availability,
        "days": days,
        "slots_per_day": slots_per_day,
        "allowed_branch_slots": allowed_branch_slots,
    }


# ---------------------------------------------------------------------------
# Scheduling helpers and constraints
# ---------------------------------------------------------------------------


BIEN_HOA_BRANCHES = {"VTS", "PVT", "TĐH", "NKH"}


@dataclass
class Session:
    day: int
    slot: int
    room: str
    teacher: str

    @property
    def time_index(self) -> int:
        return (self.day - 1) * SLOTS_PER_DAY + self.slot


class SchedulingState:
    def __init__(self, teacher_ids: List[str], days: List[int], slots_per_day: int):
        self.teacher_assignments: Dict[str, Dict[int, Dict[int, Tuple[int, str]]]] = {
            tid: {day: {} for day in days} for tid in teacher_ids
        }
        self.teacher_load: Dict[str, int] = {tid: 0 for tid in teacher_ids}
        self.room_usage: Dict[Tuple[int, int], Dict[str, int]] = {
            (day, slot): {} for day in days for slot in range(1, slots_per_day + 1)
        }

    def can_assign_teacher(
        self,
        teacher: str,
        day: int,
        slot: int,
        branch: str,
        availability: Dict[str, Dict[int, Dict[int, bool]]],
        min_max: Dict[str, Tuple[int, int]],
    ) -> bool:
        if not availability[teacher][day][slot]:
            return False
        if slot in self.teacher_assignments[teacher][day]:
            return False

        _, max_load = min_max[teacher]
        if self.teacher_load[teacher] >= max_load:
            return False

        day_slots = sorted(self.teacher_assignments[teacher][day].keys())
        # Consecutive teaching limit – prevent three in a row
        extended_slots = sorted(day_slots + [slot])
        for i in range(len(extended_slots) - 2):
            if (
                extended_slots[i + 1] - extended_slots[i] == 1
                and extended_slots[i + 2] - extended_slots[i + 1] == 1
            ):
                return False

        # Branch travel constraint for consecutive slots
        for neighbor in (slot - 1, slot + 1):
            if neighbor in self.teacher_assignments[teacher][day]:
                _, neighbor_branch = self.teacher_assignments[teacher][day][neighbor]
                if neighbor_branch != branch:
                    if (
                        branch not in BIEN_HOA_BRANCHES
                        or neighbor_branch not in BIEN_HOA_BRANCHES
                    ):
                        return False
        return True

    def assign_teacher(self, teacher: str, day: int, slot: int, class_id: int, branch: str) -> None:
        self.teacher_assignments[teacher][day][slot] = (class_id, branch)
        self.teacher_load[teacher] += 1

    def unassign_teacher(self, teacher: str, day: int, slot: int) -> None:
        if slot in self.teacher_assignments[teacher][day]:
            del self.teacher_assignments[teacher][day][slot]
            self.teacher_load[teacher] -= 1

    def can_assign_room(self, room: str, day: int, slot: int) -> bool:
        return room not in self.room_usage[(day, slot)]

    def assign_room(self, room: str, day: int, slot: int, class_id: int) -> None:
        self.room_usage[(day, slot)][room] = class_id

    def unassign_room(self, room: str, day: int, slot: int) -> None:
        if room in self.room_usage[(day, slot)]:
            del self.room_usage[(day, slot)][room]


def day_slot_pairs(days: List[int]) -> List[Tuple[int, int]]:
    pairs = []
    for d1 in days:
        for d2 in days:
            if d2 - d1 >= 2:
                pairs.append((d1, d2))
    random.shuffle(pairs)
    return pairs


def try_assign_sessions(
    class_id: int,
    sessions: List[Session],
    state: SchedulingState,
    data: Dict,
) -> bool:
    class_branch = data["class_branch"][class_id]
    class_level = data["class_level"][class_id]
    class_size = data["class_size"][class_id]
    teacher_type = data["teacher_type"]
    qualifications = data["qualifications"]
    availability = data["availability"]
    min_max = data["min_max"]
    branch_of_room = data["branch_of_room"]
    capacities = data["capacities"]

    if len(sessions) != 2:
        return False

    session_types = {teacher_type[s.teacher] for s in sessions}
    if session_types != {"full-time", "part-time"}:
        return False

    days = sorted(s.day for s in sessions)
    if days[1] - days[0] < 2:
        return False

    allocated: List[Session] = []
    for sess in sessions:
        teacher = sess.teacher
        # BranchHours: branch must be active at (day, slot)
        if (class_branch, sess.day, sess.slot) not in data["allowed_branch_slots"]:
            break
        if class_level not in qualifications.get(teacher, set()):
            break
        if not state.can_assign_teacher(teacher, sess.day, sess.slot, class_branch, availability, min_max):
            break
        if branch_of_room.get(sess.room) != class_branch:
            break
        if capacities.get(sess.room, 0) < class_size:
            break
        if not state.can_assign_room(sess.room, sess.day, sess.slot):
            break

        state.assign_teacher(teacher, sess.day, sess.slot, class_id, class_branch)
        state.assign_room(sess.room, sess.day, sess.slot, class_id)
        allocated.append(sess)
    else:
        return True

    # rollback when failed
    for sess in allocated:
        state.unassign_teacher(sess.teacher, sess.day, sess.slot)
        state.unassign_room(sess.room, sess.day, sess.slot)
    return False


def generate_sessions_for_class(
    class_id: int,
    state: SchedulingState,
    data: Dict,
    preferred: Optional[List[Session]] = None,
) -> Optional[List[Session]]:
    teacher_ids = data["teacher_ids"]
    teacher_type = data["teacher_type"]
    qualifications = data["qualifications"]
    availability = data["availability"]
    min_max = data["min_max"]
    class_branch = data["class_branch"][class_id]
    class_level = data["class_level"][class_id]
    class_size = data["class_size"][class_id]
    room_ids_per_branch = data["room_ids_per_branch"]
    capacities = data["capacities"]
    branch_of_room = data["branch_of_room"]
    days = data["days"]
    slots_per_day = data["slots_per_day"]

    full_teachers = [t for t in teacher_ids if teacher_type[t] == "full-time" and class_level in qualifications.get(t, set())]
    part_teachers = [t for t in teacher_ids if teacher_type[t] == "part-time" and class_level in qualifications.get(t, set())]
    if not full_teachers or not part_teachers:
        return None

    valid_rooms = [r for r in room_ids_per_branch.get(class_branch, []) if capacities.get(r, 0) >= class_size]
    if not valid_rooms:
        return None

    if preferred and try_assign_sessions(class_id, preferred, state, data):
        preferred_sorted = sorted(preferred, key=lambda s: (s.day, s.slot))
        return preferred_sorted

    day_pairs_list = day_slot_pairs(days)
    role_orders = [("full-time", "part-time"), ("part-time", "full-time")]
    for day1, day2 in day_pairs_list:
        for first_role, second_role in role_orders:
            first_pool = full_teachers if first_role == "full-time" else part_teachers
            second_pool = part_teachers if second_role == "part-time" else full_teachers
            if not first_pool or not second_pool:
                continue

            slot1_pool = [s for s in range(1, slots_per_day + 1)
                          if (class_branch, day1, s) in data["allowed_branch_slots"]]
            if not slot1_pool:
                continue
            for slot1 in random.sample(slot1_pool, len(slot1_pool)):
                candidate_rooms1 = [r for r in valid_rooms if state.can_assign_room(r, day1, slot1)]
                if not candidate_rooms1:
                    continue

                for teacher1 in random.sample(first_pool, len(first_pool)):
                    if not availability[teacher1][day1][slot1]:
                        continue
                    if not state.can_assign_teacher(teacher1, day1, slot1, class_branch, availability, min_max):
                        continue

                    for room1 in candidate_rooms1:
                        state.assign_teacher(teacher1, day1, slot1, class_id, class_branch)
                        state.assign_room(room1, day1, slot1, class_id)

                        success = False
                        slot2_pool = [s for s in range(1, slots_per_day + 1)
                                      if (class_branch, day2, s) in data["allowed_branch_slots"]]
                        if not slot2_pool:
                            continue
                        for slot2 in random.sample(slot2_pool, len(slot2_pool)):
                            candidate_rooms2 = [r for r in valid_rooms if state.can_assign_room(r, day2, slot2)]
                            if not candidate_rooms2:
                                continue

                            for teacher2 in random.sample(second_pool, len(second_pool)):
                                if not availability[teacher2][day2][slot2]:
                                    continue
                                if not state.can_assign_teacher(teacher2, day2, slot2, class_branch, availability, min_max):
                                    continue

                                for room2 in candidate_rooms2:
                                    state.assign_teacher(teacher2, day2, slot2, class_id, class_branch)
                                    state.assign_room(room2, day2, slot2, class_id)
                                    sessions = [
                                        Session(day1, slot1, room1, teacher1),
                                        Session(day2, slot2, room2, teacher2),
                                    ]
                                    sessions.sort(key=lambda s: (s.day, s.slot))
                                    success = True
                                    break

                                if success:
                                    break

                            if success:
                                break

                        if success:
                            return sessions

                        state.unassign_teacher(teacher1, day1, slot1)
                        state.unassign_room(room1, day1, slot1)

    return None


def build_schedule(data: Dict, preferred: Optional[Dict[int, List[Session]]] = None, max_attempts: int = 50) -> Dict[int, List[Session]]:
    class_ids = data["class_ids"]
    days = data["days"]
    slots_per_day = data["slots_per_day"]
    teacher_ids = data["teacher_ids"]

    for _ in range(max_attempts):
        state = SchedulingState(teacher_ids, days, slots_per_day)
        assignments: Dict[int, List[Session]] = {}
        order = class_ids[:]
        random.shuffle(order)

        feasible = True
        for class_id in order:
            pref_sessions = preferred.get(class_id) if preferred else None
            pref_copy = deepcopy(pref_sessions) if pref_sessions else None
            sessions = generate_sessions_for_class(class_id, state, data, pref_copy)
            if sessions is None:
                feasible = False
                break
            assignments[class_id] = sessions
        if feasible:
            return assignments

    raise RuntimeError("Unable to build a feasible schedule with the provided data and constraints.")


# ---------------------------------------------------------------------------
# Genetic algorithm components
# ---------------------------------------------------------------------------


def fitness(assignments: Dict[int, List[Session]], data: Dict) -> int:
    teacher_sessions: Dict[str, List[Tuple[int, str]]] = defaultdict(list)
    class_branch = data["class_branch"]
    for class_id, sessions in assignments.items():
        branch = class_branch[class_id]
        for session in sessions:
            teacher_sessions[session.teacher].append((session.time_index, branch))

    movements = 0
    for teacher, entries in teacher_sessions.items():
        entries.sort()
        for (t1, b1), (t2, b2) in zip(entries, entries[1:]):
            if t2 - t1 == 1 and b1 != b2:
                movements += 1
    return movements


def tournament_selection(population: List[Dict[int, List[Session]]], data: Dict, k: int = 5) -> List[Dict[int, List[Session]]]:
    selected = []
    for _ in range(len(population)):
        competitors = random.sample(population, k)
        best = min(competitors, key=lambda ind: fitness(ind, data))
        selected.append(deepcopy(best))
    return selected


def crossover(
    parent1: Dict[int, List[Session]],
    parent2: Dict[int, List[Session]],
    data: Dict,
) -> Tuple[Dict[int, List[Session]], Dict[int, List[Session]]]:
    class_ids = data["class_ids"]
    midpoint = len(class_ids) // 2
    chosen_classes = set(random.sample(class_ids, midpoint))

    preferred_child1: Dict[int, List[Session]] = {}
    preferred_child2: Dict[int, List[Session]] = {}
    for class_id in class_ids:
        if class_id in chosen_classes:
            preferred_child1[class_id] = parent1[class_id]
            preferred_child2[class_id] = parent2[class_id]
        else:
            preferred_child1[class_id] = parent2[class_id]
            preferred_child2[class_id] = parent1[class_id]

    child1 = build_schedule(data, preferred_child1)
    child2 = build_schedule(data, preferred_child2)
    return child1, child2


def mutate(individual: Dict[int, List[Session]], data: Dict, mutation_rate: float) -> Dict[int, List[Session]]:
    class_ids = data["class_ids"]
    to_mutate = [cid for cid in class_ids if random.random() < mutation_rate]
    if not to_mutate:
        to_mutate = [random.choice(class_ids)]

    preferred: Dict[int, List[Session]] = {}
    for class_id in class_ids:
        if class_id not in to_mutate:
            preferred[class_id] = individual[class_id]

    return build_schedule(data, preferred)


# ---------------------------------------------------------------------------
# Output utilities
# ---------------------------------------------------------------------------


def build_schedule_dataframe(assignments: Dict[int, List[Session]], data: Dict) -> pd.DataFrame:
    rows = []
    class_branch = data["class_branch"]
    class_level = data["class_level"]
    class_size = data["class_size"]
    teacher_type = data["teacher_type"]
    for class_id, sessions in assignments.items():
        for idx, session in enumerate(sessions, start=1):
            rows.append(
                {
                    "ClassID": class_id,
                    "Session": idx,
                    "Day": session.day,
                    "Slot": session.slot,
                    "TeacherID": session.teacher,
                    "TeacherType": teacher_type[session.teacher],
                    "RoomID": session.room,
                    "Branch": class_branch[class_id],
                    "Level": class_level[class_id],
                    "Size": class_size[class_id],
                }
            )
    return pd.DataFrame(rows)


def save_results(assignments: Dict[int, List[Session]], data: Dict, fitness_history: List[int]) -> None:
    df = build_schedule_dataframe(assignments, data)
    df.sort_values(["Day", "Slot", "Branch", "ClassID", "Session"], inplace=True)
    allowed = data.get("allowed_branch_slots", set())
    if allowed:
        bad = [(r.Branch, int(r.Day), int(r.Slot))
               for r in df.itertuples() if (r.Branch, int(r.Day), int(r.Slot)) not in allowed]
        if bad:
            raise RuntimeError(f"Schedule contains BranchHours violations: {bad[:10]} (total {len(bad)})")
    df.to_excel("schedule.xlsx", index=False)

    plt.figure()
    plt.plot(fitness_history)
    plt.xlabel("Generation")
    plt.ylabel("Fitness (movements)")
    plt.title("Fitness Evolution")
    plt.savefig("fitness.png")
    plt.close()


# ---------------------------------------------------------------------------
# Main routine
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Genetic Algorithm for Scheduling")
    parser.add_argument("--crossover_rate", type=float, default=0.8, help="Crossover probability")
    parser.add_argument("--mutation_rate", type=float, default=0.05, help="Mutation probability")
    parser.add_argument("--population_size", type=int, default=40, help="Population size")
    parser.add_argument("--generations", type=int, default=200, help="Number of generations")
    parser.add_argument(
        "--input_file",
        type=str,
        default="InputData.xlsx",
        help="Path to the Excel file containing schedule data",
    )
    # === Spyder-safe argv: allow running in Spyder without Run configuration ===
    import sys
    def _spyder_argv():
        # If running inside Spyder/spyder_kernels, return default args
        if any(k in sys.modules for k in ('spyder', 'spyder_kernels')):
            return [
                '--input_file', 'InputData.xlsx',
                '--population_size', '10',
                '--generations', '30',
                '--crossover_rate', '0.8',
                '--mutation_rate', '0.05'
            ]
        return None

    _argv = _spyder_argv()
    args = parser.parse_args(_argv if _argv is not None else None)
    # ========================================================================

    random.seed(42)

    data = load_input_data(args.input_file)

    population = [build_schedule(data) for _ in range(args.population_size)]
    fitness_history: List[int] = []

    for generation in range(args.generations):
        selected = tournament_selection(population, data)
        new_population: List[Dict[int, List[Session]]] = []
        for i in range(0, len(selected), 2):
            parent1 = selected[i]
            parent2 = selected[(i + 1) % len(selected)]
            if random.random() < args.crossover_rate:
                child1, child2 = crossover(parent1, parent2, data)
            else:
                child1 = deepcopy(parent1)
                child2 = deepcopy(parent2)

            child1 = mutate(child1, data, args.mutation_rate)
            child2 = mutate(child2, data, args.mutation_rate)
            new_population.extend([child1, child2])

        population = new_population[: args.population_size]
        best = min(population, key=lambda ind: fitness(ind, data))
        best_fit = fitness(best, data)
        fitness_history.append(best_fit)
        if generation % 10 == 0:
            print(f"Generation {generation}: Best Fitness = {best_fit}")

    best = min(population, key=lambda ind: fitness(ind, data))
    print(f"Final best fitness: {fitness(best, data)}")

    save_results(best, data, fitness_history)


if __name__ == "__main__":
    main()


import unified_planning as up
from unified_planning.shortcuts import *
import numpy as np
from timeit import default_timer as timer


class UnifiedPlanningProblem:
    def __init__(
        self,
        horizon: int,
        nb_gens: int,
        nb_storages: int,
        ptdf: list[list],
        grid_params: dict,
        init_states: dict,
    ):
        self.horizon = horizon
        self.nb_gens = nb_gens
        self.nb_storages = nb_storages
        self.ptdf = ptdf
        self.nb_buses = ptdf.shape[1]
        self.grid_params = grid_params
        self.init_states = init_states
        self.nb_transmission_lines = len(grid_params["lines"])

        start = timer()
        self._create_fluents()
        end = timer()
        print(f"Fluents creation: {end - start} seconds")
        print(
            f"Number of fluents: {len(self.pgen) + len(self.soc) + len(self.congestions) + len(self.flows)}"
        )
        start = timer()
        self._create_actions()
        end = timer()
        print(f"Actions creation: {end - start} seconds")
        print(f"Number of actions: {len(self.prod_target) + len(self.soc_target)}")
        start = timer()
        self._create_problem()
        end = timer()
        print(f"Problem creation: {end - start} seconds")

    def _create_fluents(self):
        # Creating problem 'variables' so called fluents in PDDL
        self.pgen = [
            [Fluent(f"pgen_{gen_id}_{t}", RealType()) for t in range(self.horizon)]
            for gen_id in range(self.nb_gens)
        ]

        self.soc = [
            [Fluent(f"soc_{storage_id}_{t}", RealType()) for t in range(self.horizon)]
            for storage_id in range(self.nb_storages)
        ]

        self.congestions = [
            [Fluent(f"congestion_{k}_{t}", BoolType()) for t in range(self.horizon)]
            for k in range(self.nb_transmission_lines)
        ]

        self.flows = [
            [Fluent(f"flow_{k}_{t}", RealType()) for t in range(self.horizon)]
            for k in range(self.nb_transmission_lines)
        ]

    def _create_actions(self):
        self.actions_costs = {}
        # Creating actions
        self.prod_target = []

        for gen_id in range(self.nb_gens):
            if (
                self.grid_params["gens"]["redispatchable"][gen_id] == True
                and self.grid_params["gens"]["slack"][gen_id] == False
            ):
                pmax = self.grid_params["gens"]["pmax"][gen_id]
                pmin = self.grid_params["gens"]["pmin"][gen_id]
                delta = int(pmax - pmin)
                for t in range(self.horizon - 1):
                    for i in range(delta):
                        self.prod_target.append(
                            InstantaneousAction(f"prod_target_{gen_id}_{t}_{i}")
                        )
                        action = self.prod_target[-1]
                        self.actions_costs[action] = (
                            i * self.grid_params["gens"]["gen_cost_per_MW"][gen_id]
                        )
                        action.add_precondition(
                            GE(
                                self.pgen[gen_id][t],
                                i - self.grid_params["gens"]["max_ramp_up"][gen_id],
                            )
                        )
                        action.add_precondition(
                            LE(
                                self.pgen[gen_id][t],
                                i + self.grid_params["gens"]["max_ramp_down"][gen_id],
                            )
                        )
                        action.add_effect(self.pgen[gen_id][t + 1], i)
                        for k in range(self.nb_transmission_lines):
                            action.add_increase_effect(
                                self.flows[k][t + 1],
                                float(
                                    self.ptdf[k][
                                        self.grid_params["gens"]["bus"][gen_id]
                                    ]
                                )
                                * (
                                    self.pgen[gen_id][t + 1]
                                    - float(
                                        self.init_states["gens"][gen_id]
                                    )  # TODO: add dimension t in init_states
                                ),
                            )
                            if (
                                self.flows[k][t + 1] >= 500000000000
                            ):  # TODO: add the max flow on each line in the grid_params
                                action.add_effect(self.congestions[k][t + 1], True)

        self.soc_target = []
        for storage_id in range(self.nb_storages):
            emax = self.grid_params["storages"]["Emax"][storage_id]
            emin = self.grid_params["storages"]["Emin"][storage_id]
            delta = int(emax - emin)
            for t in range(self.horizon - 1):
                for i in np.linspace(0, delta, delta * 10):
                    self.soc_target.append(
                        InstantaneousAction(f"soc_target_{storage_id}_{t}_{i}")
                    )
                    action = self.soc_target[-1]
                    self.actions_costs[action] = 0
                    action.add_precondition(
                        GE(
                            self.soc[storage_id][t],
                            round(i, 2)
                            - self.grid_params["storages"]["Emax"][storage_id],
                        )
                    )
                    action.add_precondition(
                        LE(
                            self.soc[storage_id][t],
                            round(i, 2)
                            + self.grid_params["storages"]["Emin"][storage_id],
                        )
                    )
                    action.add_effect(self.soc[storage_id][t + 1], round(i, 2))
                    # CONTINUE

    def _create_problem(self):
        problem = Problem("GridStability")

        for gen_id in range(self.nb_gens):
            for t in range(self.horizon):
                problem.add_fluent(self.pgen[gen_id][t])

        for storage_id in range(self.nb_storages):
            for t in range(self.horizon):
                problem.add_fluent(self.soc[storage_id][t])

        for k in range(self.nb_transmission_lines):
            for t in range(self.horizon):
                problem.add_fluent(self.congestions[k][t])
                problem.add_fluent(self.flows[k][t])

        problem.add_actions(self.prod_target)
        problem.add_actions(self.soc_target)

        # add initial states
        for gen_id in range(self.nb_gens):
            problem.set_initial_value(
                self.pgen[gen_id][0], float(self.init_states["gens"][gen_id])
            )

        for storage_id in range(self.nb_storages):
            problem.set_initial_value(
                self.soc[storage_id][0], float(self.init_states["storages"][storage_id])
            )

        for k in range(self.nb_transmission_lines):
            problem.set_initial_value(self.congestions[k][0], False)
            i = self.grid_params["lines"][k]["from"]
            j = self.grid_params["lines"][k]["to"]
            problem.set_initial_value(
                self.flows[k][0], float(self.init_states["flows"][i][j])
            )

        problem.add_quality_metric(
            up.model.metrics.MinimizeActionCosts(self.actions_costs)
        )

        goal = And([self.congestions[k][0] for k in range(self.nb_transmission_lines)])
        problem.add_goal(goal)

        self.problem = problem

    def solve(self):
        with Compiler(
            problem_kind=self.problem.kind, compilation_kind=CompilationKind.GROUNDING
        ) as grounder:
            grounding_result = grounder.compile(self.problem, CompilationKind.GROUNDING)
            ground_problem = grounding_result.problem
            print(ground_problem)
            # The grounding_result can be used to "lift" a ground plan back to the level of the original problem
            with OneshotPlanner(problem_kind=ground_problem.kind) as planner:
                ground_plan = planner.solve(ground_problem).plan
                print("Ground plan: %s" % ground_plan)
                # Replace the action instances of the grounded plan with their correspoding lifted version
                lifted_plan = ground_plan.replace_action_instances(
                    grounding_result.map_back_action_instance
                )
                print("Lifted plan: %s" % lifted_plan)
                # Test the problem and plan validity
                with PlanValidator(
                    problem_kind=self.problem.kind, plan_kind=ground_plan.kind
                ) as validator:
                    ground_validation = validator.validate(ground_problem, ground_plan)
                    lift_validation = validator.validate(self.problem, lifted_plan)
                    Valid = up.engines.ValidationResultStatus.VALID
                    assert ground_validation.status == Valid
                    assert lift_validation.status == Valid

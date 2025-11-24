from enum import Enum
from typing import Dict, List, Union

from app.agent.base import BaseAgent
from app.flow.base import BaseFlow
from app.flow.iterative_refinement import IterativeRefinementFlow
from app.flow.planning import PlanningFlow


class FlowType(str, Enum):
    PLANNING = "planning"
    ITERATIVE_REFINEMENT = "iterative_refinement"


class FlowFactory:
    """Factory for creating different types of flows with support for multiple agents"""

    @staticmethod
    def create_flow(
        flow_type: FlowType,
        agents: Union[BaseAgent, List[BaseAgent], Dict[str, BaseAgent]],
        **kwargs,
    ) -> BaseFlow:
        flows = {
            FlowType.PLANNING: PlanningFlow,
            FlowType.ITERATIVE_REFINEMENT: IterativeRefinementFlow,
        }

        flow_class = flows.get(flow_type)
        if not flow_class:
            raise ValueError(f"Unknown flow type: {flow_type}")

        return flow_class(agents, **kwargs)

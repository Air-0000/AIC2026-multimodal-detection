"""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

from ._solver import BaseSolver
from .clas_solver import ClasSolver
from .det_solver import DetSolver
from .dndt_solver import DNDTSolver
from .distill_solver import DistillDetSolver



from typing import Dict

TASKS :Dict[str, BaseSolver] = {
    'classification': ClasSolver,
    'detection': DetSolver,
    'distill': DistillDetSolver,
    'dndt': DNDTSolver,
}

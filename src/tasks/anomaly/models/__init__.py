from src.tasks.anomaly.models.stfpm import STFPMModel, build_stfpm
from src.tasks.anomaly.models.efficientad import EfficientAdModel, build_efficientad
from src.tasks.anomaly.models.fastflow import FastflowModel, build_fastflow
from src.tasks.anomaly.models.patchcore import PatchcoreModel, build_patchcore
from src.tasks.anomaly.models.padim import PadimModel, build_padim
from src.tasks.anomaly.models.reverse_distillation import (
    ReverseDistillationModel,
    build_reverse_distillation,
)
from src.tasks.anomaly.models.dfm import DFMModel, build_dfm
from src.tasks.anomaly.models.dfkde import DfkdeModel, build_dfkde
from src.tasks.anomaly.models.cfa import CfaModel, build_cfa
from src.tasks.anomaly.models.cflow import CflowModel, build_cflow
from src.tasks.anomaly.models.draem import DraemModel, build_draem
from src.tasks.anomaly.models.dsr import DsrModel, build_dsr
from src.tasks.anomaly.models.fre import FREModel, build_fre
from src.tasks.anomaly.models.ganomaly import GanomalyModel, build_ganomaly
from src.tasks.anomaly.models.uflow import UflowModel, build_uflow
from src.tasks.anomaly.models.csflow import CsFlowModel, build_csflow
from src.tasks.anomaly.models.supersimplenet import SupersimplenetModel, build_supersimplenet

__all__ = [
    "STFPMModel",
    "build_stfpm",
    "EfficientAdModel",
    "build_efficientad",
    "FastflowModel",
    "build_fastflow",
    "PatchcoreModel",
    "build_patchcore",
    "PadimModel",
    "build_padim",
    "ReverseDistillationModel",
    "build_reverse_distillation",
    "DFMModel",
    "build_dfm",
    "DfkdeModel",
    "build_dfkde",
    "CfaModel",
    "build_cfa",
    "CflowModel",
    "build_cflow",
    "DraemModel",
    "build_draem",
    "DsrModel",
    "build_dsr",
    "FREModel",
    "build_fre",
    "GanomalyModel",
    "build_ganomaly",
    "UflowModel",
    "build_uflow",
    "CsFlowModel",
    "build_csflow",
    "SupersimplenetModel",
    "build_supersimplenet",
]

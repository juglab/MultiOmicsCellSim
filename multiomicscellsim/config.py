from pydantic import BaseModel, Field, validator
from typing import List, Literal

class TissueEuclideanConfig(BaseModel):

    # Guidelines
    n_curves: int = Field(2, description="Number of different guidelines to grow cells onto")
    curve_types: Literal["circles"] = Field("circles", description="Kind of guidelines to use")
    variances: List[float] = Field([1., 1.], description="List of variances for each guideline")

    @validator('variances')
    def check_variance_length(cls, v, values):
        n_curves = values.get('n_curves')
        if n_curves is not None and len(v) != n_curves:
            raise ValueError(f"Variance list must have the same length as n_curves ({n_curves}).")
        return v


class CellEuclideanConfig(BaseModel):
    pass

class SubcellularEuclideanConfig(BaseModel):
    pass


class SimulatorConfig(BaseModel):
    log_level: Literal["INFO", "WARNING", "DEBUG", "ERROR"] = Field("DEBUG", description="Logging Level")
    tissue_config: TissueEuclideanConfig = Field(TissueEuclideanConfig(), description="Tissue-level input features for euclidean space")
    cell_config: CellEuclideanConfig = Field(CellEuclideanConfig(), description="Cell-level input features for euclidean space")
    subcellular_config: SubcellularEuclideanConfig = Field(SubcellularEuclideanConfig(), description="Subcellular-level input features for euclidean space")
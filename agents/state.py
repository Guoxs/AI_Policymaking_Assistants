from pydantic import BaseModel, Field
from typing import Optional, TypedDict, List, Dict, Any



class PopulationState(BaseModel):
    """Container for population state in SEIR compartments."""
    
    susceptible: int = 0    # people who is not infected yet but can be infected
    exposed: int = 0        # people who is infected but not infectious yet
    infected: int = 0       # people who is infected and infectious
    confirmed: int = 0     # people who is confirmed and quarantined
    recovered: int = 0      # people who is recovered and immune
    deaths: int = 0         # people who is dead
    Npop: int = 0           # total population, should be constant
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            'S': self.susceptible,
            'E': self.exposed,
            'I': self.infected,
            'Q': self.confirmed,
            'R': self.recovered,
            'D': self.deaths
        }
    
    def total_population(self) -> Any:
        """Calculate total population."""
        return self.susceptible + self.exposed + self.infected + self.confirmed + self.recovered + self.deaths
    

class EpidemicParameters(BaseModel):
    """Container for epidemic model parameters."""
    
    # Basic SEIR parameters
    beta: float = 0.5      # Transmission rate, susceptible -> exposed
    alpha: float = 0.2     # Incubation rate, exposed -> infected
    gamma: float = 0.1     # Recovery rate, infected -> recovered
    mu: float = 0.0        # Death rate, infected -> deaths

class EpidemicParameters_v2(BaseModel):
    """Container for epidemic model parameters with initial conditions."""
    # Basic SEIR parameters
    beta: float = 0.5      # Transmission rate, susceptible -> exposed
    alpha: float = 0.2     # Incubation rate, exposed -> infected——
    gamma: float = 0.1     # Recovery rate, infected -> recovered
    mu: float = 0.0        # Death rate, infected -> deaths
    delta: float = 0.1    # Confirmation rate, infected -> confirmed
    beta_2: float = 0.0   # Secondary transmission rate
    delta_2: float = 0.0  # Secondary confirmation rate
    gamma_params: Optional[List[float]] = None
    mu_params: Optional[List[float]] = None


class TransportationState(BaseModel):
    """Container for transportation data between regions."""
    inflow: Dict[str, int] = Field(default_factory=dict, description="Inflow from neighboring regions")
    outflow: Dict[str, int] = Field(default_factory=dict, description="Outflow to neighboring regions")


class PolicyResponse(BaseModel):
    """Response structure for policy decisions."""
    policy: Dict[str, float] = Field(default_factory=dict)  # Mapping of region_id to reduction_ratio
    explanation: str = ""  # Explanation for the policy decisions

class PolicyResponse_v2(BaseModel):
    """Response structure for policy decisions."""
    policy: Dict[str, list] = Field(default_factory=dict)  # Mapping of region_id to reduction_ratio
    explanation: str = ""  # Explanation for the policy decisions

# class PolicyResponse_v3(BaseModel):
#     ## Response structure for restriction policy
#     policy: Dict[str, str] = Field(default_factory=dict)  # Mapping of region_id to restricted_origin state or null
#     explanation: str = ""  # Explanation for the policy decisions

class RegionState(BaseModel):
    region_id: str = Field("New York", description="Unique identifier for the region")
    neighboring_region_ids: List[str] = Field([], description="List of neighboring region IDs")

    epidemic_params: EpidemicParameters_v2 = Field(default_factory=EpidemicParameters_v2, description="Epidemic model parameters for the region")
    epidemic_inspect: List[EpidemicParameters_v2] = Field([], description="History of fitted epidemic parameters for inspection")
    
    current_population: PopulationState = Field(default_factory=PopulationState, description="current simulated population state for the region")
    current_mobility: TransportationState = Field(default_factory=TransportationState, description="Current mobility data for the region")
    
    population_history: List[PopulationState] = Field(None, description="History of simulated population states")
    mobility_history: List[TransportationState] = Field(None, description="History of mobility data")
    
    policy_response: PolicyResponse = Field(None, description="Latest policy response from the region agent")
    
    gt_populations: List[PopulationState] = Field([], description="Ground truth population state for the region")
    gt_mobilities: List[TransportationState] = Field([], description="Ground truth mobility data for the region")


class WorkflowState(TypedDict):
    """Represents the current state of the workflow."""
    max_iterations: int = Field(10, description="total iteration steps")
    current_iteration_step: int = Field(0, description="current iteration step")
    simulation_steps: int = Field(7, description="number of days to simulate ahead each iteration")
    region_ids: List[str] = Field([], description="List of all region IDs")
    region_infos: Dict[str, RegionState] = Field({}, description="Mapping of region_id to RegionState")   
    
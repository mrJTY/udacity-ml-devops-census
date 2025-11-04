from pydantic import BaseModel, Field
from typing import Literal


class ModelRequest(BaseModel):
    """
    Request schema for census income prediction.
    """
    age: int = Field(..., ge=17, le=90, description="Age in years")
    workclass: Literal[
        "state_gov", "self_emp_not_inc", "private", "federal_gov",
        "local_gov", "self_emp_inc", "without_pay", "never_worked"
    ] = Field(..., description="Type of employment")
    fnlgt: int = Field(..., description="Final sampling weight")
    education: Literal[
        "bachelors", "hs_grad", "11th", "masters", "9th", "some_college",
        "assoc_acdm", "assoc_voc", "7th_8th", "doctorate", "prof_school",
        "5th_6th", "10th", "1st_4th", "preschool", "12th"
    ] = Field(..., description="Highest education level")
    education_num: int = Field(..., ge=1, le=16, description="Education in numerical form")
    marital_status: Literal[
        "never_married", "married_civ_spouse", "divorced",
        "married_spouse_absent", "separated", "married_af_spouse", "widowed"
    ] = Field(..., description="Marital status")
    occupation: Literal[
        "adm_clerical", "exec_managerial", "handlers_cleaners",
        "prof_specialty", "other_service", "sales", "craft_repair",
        "transport_moving", "farming_fishing", "machine_op_inspct",
        "tech_support", "protective_serv", "armed_forces", "priv_house_serv"
    ] = Field(..., description="Occupation type")
    relationship: Literal[
        "not_in_family", "husband", "wife", "own_child", "unmarried", "other_relative"
    ] = Field(..., description="Family relationship")
    race: Literal[
        "white", "black", "asian_pac_islander", "amer_indian_eskimo", "other"
    ] = Field(..., description="Race")
    sex: Literal["male", "female"] = Field(..., description="Gender")
    capital_gain: int = Field(..., ge=0, description="Capital gains")
    capital_loss: int = Field(..., ge=0, description="Capital losses")
    hours_per_week: int = Field(..., ge=1, le=99, description="Hours worked per week")
    native_country: str = Field(..., description="Country of origin")

    class Config:
        json_schema_extra = {
            "example": {
                "age": 39,
                "workclass": "state_gov",
                "fnlgt": 77516,
                "education": "bachelors",
                "education_num": 13,
                "marital_status": "never_married",
                "occupation": "adm_clerical",
                "relationship": "not_in_family",
                "race": "white",
                "sex": "male",
                "capital_gain": 2174,
                "capital_loss": 0,
                "hours_per_week": 40,
                "native_country": "united_states"
            }
        }


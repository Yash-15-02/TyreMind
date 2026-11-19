"""Validated handoff and initial investigation state; no LLM execution."""
import hashlib
import json
from typing import Any,Literal

from pydantic import BaseModel,ConfigDict,Field,AwareDatetime,model_validator


class StrictModel(BaseModel):
    model_config=ConfigDict(extra='forbid')


class Job(StrictModel):
    investigation_id:str=Field(pattern=r'^[A-Za-z0-9_-]{1,128}$')
    event_id:str=Field(min_length=1)
    target_revision:int=Field(ge=1)
    priority:Literal['High','Critical']
    requested_assessment_id:str=Field(min_length=1)


class Event(StrictModel):
    event_id:str
    tyre_id:str=Field(min_length=1)
    truck_id:str=Field(min_length=1)
    status:Literal['OPEN']
    current_level:Literal['Normal','Medium','High','Critical']|None
    highest_level:Literal['High','Critical']
    first_seen:AwareDatetime
    last_seen:AwareDatetime
    revision:int=Field(ge=1)


class Assessment(StrictModel):
    assessment_id:str
    timestamp:AwareDatetime
    feature_as_of:AwareDatetime
    risk_level:Literal['High','Critical']
    data_status:Literal['complete','partial']
    policy_version:str
    policy_sha256:str
    prediction_id:str|None


class Handoff(StrictModel):
    schema_version:Literal['1.0']
    message_type:Literal['INVESTIGATE_TYRE']
    handoff_id:str
    investigation:Job
    event:Event
    assessment:Assessment
    triggered_rules:list[dict[str,Any]]=Field(min_length=1)
    data_issues:list[str]
    evidence:dict[str,Any]

    @model_validator(mode='after')
    def consistent_references(self):
        if self.investigation.event_id!=self.event.event_id:
            raise ValueError('Job/event identity mismatch.')
        if self.investigation.requested_assessment_id!=self.assessment.assessment_id:
            raise ValueError('Requested assessment mismatch.')
        if self.investigation.target_revision!=self.event.revision:
            raise ValueError('Handoff revision mismatch.')
        for key in ('tyre_id','truck_id'):
            if key in self.evidence and self.evidence[key]!=getattr(self.event,key):
                raise ValueError('Evidence identity mismatch.')
        if self.assessment.timestamp>self.assessment.feature_as_of:
            raise ValueError('Assessment time exceeds its evidence cutoff.')
        return self


def validate_handoff(payload):
    message=Handoff.model_validate(payload)
    base={key:value for key,value in payload.items() if key!='handoff_id'}
    expected=hashlib.sha256(json.dumps(base,sort_keys=True,allow_nan=False).encode()).hexdigest()
    if expected!=message.handoff_id:
        raise ValueError('Handoff identity does not match its payload.')
    return message


class InvestigationState(StrictModel):
    schema_version:Literal['1.0']='1.0'
    stage:Literal['RECEIVED']='RECEIVED'
    investigation_id:str
    event_id:str
    tyre_id:str
    truck_id:str
    handoff_id:str
    target_revision:int
    priority:Literal['High','Critical']
    risk_level:Literal['High','Critical']
    received_at:AwareDatetime
    source_handoff:dict[str,Any]
    telemetry:dict[str,Any]=Field(default_factory=dict)
    ml_prediction:dict[str,Any]|None=None
    health_assessment:dict[str,Any]=Field(default_factory=dict)
    root_cause_hypotheses:list[dict[str,Any]]=Field(default_factory=list)
    retrieved_evidence:list[dict[str,Any]]=Field(default_factory=list)
    maintenance_plan:dict[str,Any]=Field(default_factory=dict)
    errors:list[str]=Field(default_factory=list)

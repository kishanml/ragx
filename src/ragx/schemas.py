from typing import List, Literal
from pydantic import BaseModel, Field


class Chunk(BaseModel):
    
    id: str = ""
    title : str = ""
    source : str = ""
    cleaned_text : str = ""
    summary: str = ""
    keywords: List[str] = None  
    questions: List[str] = None   


class Document(BaseModel):
    
    id: str = ""
    name: str = ""
    source : str = ""
    page_count: int  = ""  
    category: Literal["SOP","Manual","Untitled"] = "Untitled"
    process_name: str = ""
    chunks : List[Chunk] = None



class Output(BaseModel):
    
    summary : str = Field(...,description="Summary of the text chunk")
    questions : list[str] = Field(...,description="Set of questions that the chunk can answer.")
    keywords : list[str] = Field(..., description="Keywords from text")
    
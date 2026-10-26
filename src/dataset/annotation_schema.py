"""
Annotation Schema for Multimodal Comic Translation Dataset
Grounded in Martinec & Salway (2005) and Lommel et al. (2014) MQM.
"""

from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


class TextImageRelation(str, Enum):
    NECESSARY = "necessary"      # Translation cannot be correctly resolved without image
    HELPFUL = "helpful"          # Image provides helpful stylistic, emotional, or scene cues
    REDUNDANT = "redundant"      # Text is completely self-contained and unambiguous
    IRRELEVANT = "irrelevant"    # Dialogue is unrelated to the visual action


class LinguisticPhenomenon(str, Enum):
    GENDER = "gender"                  # Ambiguous pronoun/referent resolved by visual appearance
    NUMBER = "number"                  # Plural/singular disambiguated visually
    ADDRESS_FORM = "address_form"      # Formal (usted/vous) vs informal (tu) based on social hierarchy/age
    DEIXIS = "deixis"                  # Spatial pointing ("this", "there", "that guy")
    ONOMATOPOEIA = "onomatopoeia"      # Comic sound effect tied to visual action
    IDIOM_PUN = "idiom_pun"            # Multimodal wordplay involving both picture and word
    ELLIPSIS = "ellipsis"              # Elliptical dialogue completed by visual action
    CULTURE = "culture"                # Cultural visual/textual reference
    NONE = "none"


class MQMSeverity(str, Enum):
    MINOR = "minor"         # Penalty: 1
    MAJOR = "major"         # Penalty: 5
    CRITICAL = "critical"   # Penalty: 10


class MQMError(BaseModel):
    category: str = Field(..., description="e.g. Accuracy/Addition, Accuracy/Mistranslation, Fluency/Grammar")
    severity: MQMSeverity
    span: Optional[str] = None
    is_image_dependent: bool = Field(False, description="True if caused by ignoring or misinterpreting visual context")
    explanation: Optional[str] = None


class ComicDialogueUnit(BaseModel):
    unit_id: str = Field(..., description="Unique ID: {comic_id}_p{page}_pan{panel}_b{bubble}")
    comic_title: str
    issue_number: Optional[str] = None
    page_number: int
    panel_index: int
    bubble_index: int
    reading_order: int
    
    # Text
    gold_transcription: str
    previous_context: Optional[str] = ""
    next_context: Optional[str] = ""
    
    # Coordinates (normalized 0.0 - 1.0 or pixel coordinates)
    bubble_bbox: List[float] = Field(..., description="[ymin, xmin, ymax, xmax]")
    panel_bbox: List[float] = Field(..., description="[ymin, xmin, ymax, xmax]")
    
    # Relative Paths
    panel_image_path: str
    bubble_image_path: Optional[str] = None
    
    # Grounding Labels
    text_image_relation: TextImageRelation
    phenomena: List[LinguisticPhenomenon] = Field(default_factory=list)
    
    # Human Reference Translation (Optional gold target)
    reference_es: Optional[str] = None
    reference_fr: Optional[str] = None
    
    # Evaluation Annotation
    mqm_annotations: Optional[List[MQMError]] = None

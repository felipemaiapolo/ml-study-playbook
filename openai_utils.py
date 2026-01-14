from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field
from openai import OpenAI

client = OpenAI(api_key="")


class NegSentWithExplanation(BaseModel):
  explanation: str = Field(..., description="Brief rationale for the estimate.")
  p_negative: float = Field(..., ge=0.0, le=1.0, description="P(sentiment is negative) in [0,1].")


def QueryOAI(
  text: str,
  *,
  model: str = "gpt-5-nano-2025-08-07"
) -> NegSentWithExplanation:
  response = client.responses.parse(
    model=model,
    input=[
      {
        "role": "system",
        "content": (
          "Provide a brief explanation, then a calibrated probability p_negative in [0,1] "
          "for P(sentiment is negative). Keep the explanation short."
        )
      },
      {"role": "user", "content": text}
    ],
    text_format=NegSentWithExplanation
  )
  return response.output_parsed
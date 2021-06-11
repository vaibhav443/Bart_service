from transformers import pipeline

classifier = pipeline("zero-shot-classification",
                      model="facebook/bart-large-mnli")

from typing import List
from fastapi import FastAPI
from pydantic import BaseModel


class Payload(BaseModel):
    text: str
    keyword: List[str]


app = FastAPI()


@app.post("/bart")
def bart_service(input: Payload):
    """
    Given a piece of keyword, and text, it returns a score.
    params:
            text: string,
            keyword: list with one element in case of grading
             (multiple elements in case of topic classification)
    """

    return classifier(input.text, input.keyword)

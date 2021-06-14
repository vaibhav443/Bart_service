from transformers import pipeline

classifier1 = pipeline("zero-shot-classification",
                       model="facebook/bart-large-mnli")

classifier2 = pipeline("zero-shot-classification",
                       model="joeddav/xlm-roberta-large-xnli")
from typing import List
from fastapi import FastAPI
from pydantic import BaseModel


class Payload(BaseModel):
    text: str
    keyword: List[str]
    lang: str  # 'en' for english, 'fr' for french etc...


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
    if (input.lang == 'en'):
        return classifier1(input.text, input.keyword)
    else:
        return classifier2(input.text, input.keyword)

import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from datasets import Dataset
import os
from rag_chain import build_chain
from langchain_groq import ChatGroq
from ragas.llms import LangchainLLMWrapper
from langchain_huggingface import HuggingFaceEmbeddings
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
# Groq supports n=1, so generate one question per answer
answer_relevancy.strictness = 1
# 1 Confuguration 
load_dotenv()

if not os.getenv("GROQ_API_KEY"):
    raise ValueError("GROQ_API_KEY is missing from your .env file.")

BASE_DIR = Path(__file__).resolve().parent

INPUT_CSV=BASE_DIR/"telecom_rag_evaluation_dataset.csv"
OUTPUT_DIR=BASE_DIR/"evaluation_results"
OUTPUT_DIR.mkdir(exist_ok=True)

RAW_RESULTS = OUTPUT_DIR / "rag_outputs.csv"
METRICS_RESULTS = OUTPUT_DIR / "ragas_scores.csv"

MAX_QUESTIONS=3
#load the evaluation dataset

df =pd.read_csv(INPUT_CSV)

required_columns = [
    "question",
    "ground_truth",
    "reference_context",
]
missing = set(required_columns) - set(df.columns)

if missing:
    raise ValueError(f"Missing csv cols:{missing}")

if MAX_QUESTIONS is not None:
    df = df.head(MAX_QUESTIONS)

df = df.fillna("")

print(f"Loaded {len(df)} evaluation questions.")

#initialiing the chain 
chain=build_chain()
answers=[]
contexts_list=[]
sources_list=[]
errors=[]

#runnig every question 
for index,row in df.iterrows():
    question=str(row["question"])
    print(
        f"\nQuestion {len(answers) + len(errors) + 1}"
        f"/{len(df)}: {question}"
    )
    try:
        result=chain.invoke(question)
        answer=result["answer"]
        contexts = result["contexts"]
        sources = result.get("sources", [])
        answers.append(answer)
        contexts_list.append(list(contexts))
        sources_list.append(list(sources))
        errors.append("")
        print("Answer Generated successfully")
    except Exception as exc:
        # Keep a record of failed questions.
        answers.append("")
        contexts_list.append([])
        sources_list.append([])
        errors.append(str(exc))

        print(f"ERROR: {exc}")
#5 saving the pipe lines
raw_df = df.copy()

raw_df["generated_answer"] = answers
raw_df["retrieved_contexts"] = [
    "\n\n---\n\n".join(contexts)
    for contexts in contexts_list
]
raw_df["retrieved_sources"] = [
    ", ".join(sources)
    for sources in sources_list
]
raw_df["execution_error"] = errors

raw_df.to_csv(RAW_RESULTS, index=False, encoding="utf-8-sig")

print(f"\nRaw results saved to: {RAW_RESULTS}")

# --------------------------------------------------
# 6. Prepare successful rows for Ragas
# --------------------------------------------------

valid_indices = [
    i for i, error in enumerate(errors)
    if not error and answers[i].strip() and contexts_list[i]
]

if not valid_indices:
    raise RuntimeError(
        "No successful rows with answers and contexts. "
        "Check the raw results CSV."
    )

eval_data = {
    "question": [
        str(df.iloc[i]["question"])
        for i in valid_indices
    ],
    "answer": [
        answers[i]
        for i in valid_indices
    ],
    "contexts": [
        contexts_list[i]
        for i in valid_indices
    ],
    "ground_truth": [
        str(df.iloc[i]["ground_truth"])
        for i in valid_indices
    ],
}

dataset = Dataset.from_dict(eval_data)
#7 llm config judgng 

judge_model = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
)

judge_llm = LangchainLLMWrapper(judge_model)

metrics = [
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
]

print("\nStarting Ragas evaluation...")

evaluation_embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

result = evaluate(
    dataset=dataset,
    metrics=metrics,
    llm=judge_llm,
    embeddings=evaluation_embeddings,
    raise_exceptions=True,
    max_workers=1,
)
#saving
scores_df = result.to_pandas()

scores_df.to_csv(
    METRICS_RESULTS,
    index=False,
    encoding="utf-8-sig",
)

print("\nEvaluation complete!")
print(scores_df.mean(numeric_only=True))

print(f"\nDetailed scores saved to: {METRICS_RESULTS}")
print(f"Raw outputs: {RAW_RESULTS}")


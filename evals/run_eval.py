import json
from llamaindex_version import query_engine

questions = json.load(open("evals/eval_questions.json"))

for q in questions:
    resp = query_engine.query(q["question"])
    q["answer"] = str(resp)
    q["contexts"] = [n.text for n in resp.source_nodes]

json.dump(questions, open("evals/eval_results.json", "w"), indent=2)

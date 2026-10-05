"""radkev.external: sentence splitting (RadCases hashes depend on it) and RadGraph-XL status questions. Offline."""
from radkev import external as be


def test_first_sentence_and_hash_are_stable():
    s = "A 45-year-old man presents with chest pain. He has a history of hypertension."
    assert be.split_into_sentences(s)[0] == "A 45-year-old man presents with chest pain."
    assert be.sha512("abc").startswith("ddaf35a193617aba")


def test_radgraph_status_questions_never_key_not_mentioned():
    rows = [{"dataset": "stanford-chest-ct", "doc_key": "1",
             "sentences": [["No", "pleural", "effusion", "."], ["Small", "nodule", "in", "the", "right", "lung", "."], ["Possible", "atelectasis", "."], ["Heart", "normal", "."]],
             "ner": [[[1, 2, "Observation::definitely absent"]], [[9, 9, "Anatomy::definitely present"], [5, 5, "Observation::definitely present"]],
                     [[12, 12, "Observation::uncertain"]], [[15, 15, "Observation::definitely present"]]]}]
    docs = list(be.dygie_docs(rows, lambda d: "chestct"))
    recs = list(be.radgraph_records(docs))
    labels = sorted(q["label"] for r in recs for q in r["questions"].values())
    assert labels == ["absent", "present", "uncertain"]
    assert all("normal" not in q["instructions"] for r in recs for q in r["questions"].values())   # generic qualifier skipped
    assert all(set(q["criteria"]) == {"present", "absent", "uncertain", "not_mentioned"} for r in recs for q in r["questions"].values())

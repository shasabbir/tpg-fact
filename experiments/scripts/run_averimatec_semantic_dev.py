"""Use the pinned AVerImaTeC evidence scorer on frozen development selections."""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

from export_averimatec_outputs import _attach_frozen_verdicts
from trace_scv.data.averimatec import load_annotations
from trace_scv.data.io import sha256_file
from trace_scv.external_output import build_official_predictions
from trace_scv.external_study import read_jsonl


MODEL_REVISION = "005ad3404e59d6023443cb575daa05336842228a"
ORGANIZER_REVISION = "f6f361d2fc5a0ab7d3ea282e2beced56b928e1bd"
VALIDATION_SHA256 = "e9c61d6538acc8e8312cf3137ddf8e06efe2a8a2a5ab60583c56344ae8e81166"
POLICIES = ("fixed", "calibrated_threshold", "expected_f1",
            "uncalibrated_expected_f1", "completeness_aware_switch")


def digest(value):
    data = json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path, value):
    temporary = path.with_suffix(".part")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def prepare(experiments, root):
    validation = experiments / "data/public/averimatec/raw/val.json"
    if sha256_file(validation) != VALIDATION_SHA256:
        raise ValueError("Validation annotation hash mismatch")
    summary = load_json(root / "locked_development/summary.json")
    paths = {
        "claim_records": root / "locked_development/claim_records.jsonl",
        "development_candidate_records": root / "prepared_validation/candidate_records.jsonl",
        "train_only_freeze": root / "train_only.freeze.json",
    }
    hashes = {key: sha256_file(path) for key, path in paths.items()}
    if any(value != summary["artifacts"][key]["sha256"] for key, value in hashes.items()):
        raise ValueError("Frozen input hash mismatch")
    claims = load_annotations(validation, split="validation")
    candidates = read_jsonl(paths["development_candidate_records"])
    records = read_jsonl(paths["claim_records"])
    freeze = load_json(paths["train_only_freeze"])
    if any(row["split"] != "validation" for row in candidates + records):
        raise ValueError("Unexpected split in frozen inputs")
    ids = {row["claim_id"] for row in claims}
    jobs = {}
    for seed in (entry["seed"] for entry in summary["seeds"]):
        for policy in POLICIES:
            rows = [row for row in records if row["seed"] == seed and row["policy"] == policy]
            if len(rows) != len(ids) or {row["claim_id"] for row in rows} != ids:
                raise ValueError(f"Incomplete prediction set for {seed}/{policy}")
            if any("predicted_verdict" not in row for row in rows):
                rows = _attach_frozen_verdicts(candidates, rows, freeze, seed=seed)
            predictions, _ = build_official_predictions(claims, candidates, rows)
            jobs[f"{seed}/{policy}"] = predictions

    image_root = validation.parent / "claim_images/images"
    if not image_root.is_dir():
        raise FileNotFoundError(image_root)
    references = {}
    for claim in claims:
        names = {name for q in claim["questions"] for name in q["input_images"]}
        names.update(name for q in claim["questions"] for a in q["answers"]
                     if a["answer_type"] == "Image" for name in a["image_answers"])
        for name in sorted(names):
            if Path(name).name != name or not (image_root / name).is_file():
                raise FileNotFoundError(f"Reference image missing or invalid: {name}")
            references[name] = image_root / name
    predicted_names = {name for predictions in jobs.values() for row in predictions
                       for item in row["evidence"] for name in item["images"]}
    allowed = (validation.parent / "knowledge_store/extracted/validation").resolve()
    predicted_images = {}
    for name in sorted(predicted_names):
        path = (experiments / name).resolve()
        if not path.is_relative_to(allowed) or not path.is_file():
            raise FileNotFoundError(f"Prediction image missing or outside validation store: {name}")
        predicted_images[name] = path
    return claims, jobs, references, predicted_images, hashes


def choose_pilot(claims, jobs):
    """Pick low-cost text and image cases from inputs before scoring."""
    options = sorted(range(len(claims)), key=lambda i: (len(claims[i]["questions"]), i))
    has_reference_image = {
        i: any(q["input_images"] or any(a["answer_type"] == "Image" for a in q["answers"])
               for q in claims[i]["questions"]) for i in options
    }
    text = next((i, key) for i in options for key, rows in jobs.items()
                if all(not item["images"] for item in rows[i]["evidence"])
                and not has_reference_image[i])
    image = next(((i, key) for i in options for key, rows in jobs.items()
                 if any(item["images"] for item in rows[i]["evidence"])
                 and has_reference_image[i]), None)
    if image is None:
        image = next((i, key) for i in options for key, rows in jobs.items()
                     if has_reference_image[i] and i != text[0])
    return [text, image]


def organizer_imports(organizer):
    source_paths = ["prepare_submission", "templates"]
    revision = subprocess.check_output(["git", "-C", organizer, "rev-parse", "HEAD"],
                                       text=True).strip()
    if revision != ORGANIZER_REVISION:
        raise ValueError("Organizer repository revision mismatch")
    subprocess.run(["git", "-C", organizer, "diff", "--quiet", "HEAD", "--", *source_paths],
                   check=True)
    os.chdir(organizer / "prepare_submission")
    sys.path[:0] = [str(Path.cwd()), str(organizer)]
    # Organizer modules find the prompt files relative to prepare_submission.
    modules = tuple(importlib.import_module(name) for name in ("ref_eval", "utils"))
    if any(not Path(module.__file__).resolve().is_relative_to(organizer) for module in modules):
        raise ImportError("Organizer module resolved outside the pinned source")
    return modules


class TimedModel:
    def __init__(self, model):
        self.model = model
        self.calls = []

    @property
    def device(self):
        return self.model.device

    def generate(self, **kwargs):
        start = time.perf_counter()
        out = self.model.generate(**kwargs)
        self.calls.append({"seconds": time.perf_counter() - start,
                           "input_tokens": int(kwargs["input_ids"].shape[-1]),
                           "output_tokens": int(out.shape[-1] - kwargs["input_ids"].shape[-1]),
                           "image_input": "pixel_values" in kwargs})
        print(f"  model call {len(self.calls)}: {self.calls[-1]}", file=sys.stderr, flush=True)
        return out


def image_ratings(ref_eval, judge, evidence, reference, parsed):
    # The upstream scorer catches runtime errors and assigns a perfect image
    # score. Audit caught errors in its frame so infrastructure faults fail loudly.
    errors = []
    code = ref_eval.compute_image_scores.__code__

    def trace(frame, event, arg):
        if frame.f_code is code:
            if event == "exception":
                errors.append(type(arg[1]).__name__)
            return trace
        return None

    old_trace = sys.gettrace()
    log = io.StringIO()
    try:
        sys.settrace(trace)
        with redirect_stdout(log):
            ratings = ref_eval.compute_image_scores(judge, "gemma", evidence, reference, parsed)
    finally:
        sys.settrace(old_trace)
    if errors:
        raise RuntimeError(f"Organizer image scoring caught exceptions: {errors}")
    return ratings


def recall_by_rule(utils, response, ratings, references, predictions):
    precision, recall, f1 = utils.get_auto_recall(response, ratings, references, predictions)
    if recall is None:
        raise ValueError("Undefined organizer recall")
    code_threshold = utils.threshold
    try:
        utils.threshold = 8
        _, paper_recall, _ = utils.get_auto_recall(response, ratings, references, predictions)
    finally:
        utils.threshold = code_threshold
    return precision, recall, f1, paper_recall


def run_scoring(args, claims, jobs, references, predicted_images, hashes, cases):
    import torch
    import transformers
    from transformers import AutoProcessor, Gemma3ForConditionalGeneration, set_seed

    if args.model_dir is None or args.model_dir.name != MODEL_REVISION:
        raise ValueError(f"Pass the local snapshot directory for {MODEL_REVISION}")
    snapshot = args.model_dir
    index = load_json(snapshot / "model.safetensors.index.json")
    shards = sorted(set(index["weight_map"].values()))
    weight_hashes = {name: sha256_file(snapshot / name) for name in shards}
    model_files = {p.name: sha256_file(p) for p in snapshot.iterdir()
                   if p.is_file() and p.suffix in {".json", ".model"}}
    manifest = {
        "split": "validation", "mode": args.mode, "test_annotations_accessed": False,
        "organizer_revision": ORGANIZER_REVISION, "model_revision": MODEL_REVISION,
        "checkpoint_files": weight_hashes | model_files, "frozen_inputs": hashes,
        "validation_hash": VALIDATION_SHA256, "precision": "bfloat16",
        "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
        "transformers": transformers.__version__,
        "memory_limits": {"gpu_gib": args.gpu_gib, "cpu_gib": args.cpu_gib},
        "rng": "deterministic seed per reference and unique ordered prediction",
        "cases_sha256": digest(cases), "case_count": len(cases),
        "thresholds": {"pinned_code_image": 9, "paper_image": 8,
                       "notebook_verdict_strictly_above": 0.4, "paper_verdict_minimum": 0.3},
    }
    manifest_path = args.output / "manifest.json"
    if manifest_path.exists() and load_json(manifest_path) != manifest:
        raise ValueError("Resume inputs or runtime have changed")
    atomic_json(manifest_path, manifest)

    ref_eval, utils = organizer_imports(args.organizer)
    if utils.threshold != manifest["thresholds"]["pinned_code_image"]:
        raise ValueError("Unexpected image threshold in pinned organizer source")
    image_stage = args.output / "ref_assets/data/data_clean/images"
    image_stage.mkdir(parents=True, exist_ok=True)
    for name, source in references.items():
        target = image_stage / name
        if not target.exists():
            os.link(source, target)
        elif not target.samefile(source):
            raise ValueError(f"Reference staging mismatch for {name}")

    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    started = time.perf_counter()
    model = Gemma3ForConditionalGeneration.from_pretrained(
        snapshot, local_files_only=True, torch_dtype=torch.bfloat16, device_map="auto",
        max_memory={0: f"{args.gpu_gib}GiB", "cpu": f"{args.cpu_gib}GiB"},
        offload_folder=str(args.output / "offload"),
    ).eval()
    judge = {"model": TimedModel(model), "processor": AutoProcessor.from_pretrained(
        snapshot, local_files_only=True)}
    device_map = dict(model.hf_device_map)
    atomic_json(args.output / "runtime.json", {
        "load_seconds": time.perf_counter() - started, "device_map": device_map,
        "do_sample": model.generation_config.do_sample,
        "top_p": model.generation_config.top_p, "top_k": model.generation_config.top_k,
    })
    print(f"Checkpoint loaded in {time.perf_counter() - started:.1f}s; device placement "
          f"{sorted(set(map(str, device_map.values())))}", flush=True)

    scores = []
    for index, key in cases:
        claim, prediction = claims[index], jobs[key][index]
        claim_id = claim["claim_id"]
        reference_path = args.output / f"reference-{claim_id}.json"
        if reference_path.exists():
            reference = load_json(reference_path)["evidence"]
        else:
            set_seed(int(digest(["reference", claim_id])[:8], 16))
            before = len(judge["model"].calls)
            reference = [utils.convert_qa_format(q, judge, "gemma",
                         str(args.output / "ref_assets")) for q in claim["questions"]]
            atomic_json(reference_path, {"evidence": reference,
                        "calls": judge["model"].calls[before:]})
        identity = digest([claim_id, prediction["evidence"]])
        score_path = args.output / f"score-{identity}.json"
        if score_path.exists():
            result = load_json(score_path)
        else:
            set_seed(int(identity[:8], 16))
            before = len(judge["model"].calls)
            evidence = [{"text": item["text"],
                         "images": [str(predicted_images[p]) for p in item["images"]]}
                        for item in prediction["evidence"]]
            response, parsed = ref_eval.val_evid_idv(judge, "gemma", evidence, reference, False, True)
            ratings = image_ratings(ref_eval, judge, evidence, reference, parsed)
            precision, recall, f1, paper_recall = recall_by_rule(
                utils, response, ratings, len(reference), len(evidence))
            result = {"precision": precision, "recall": recall, "f1": f1,
                      "paper_rule_recall": paper_recall,
                      "judge_response": response, "parsed": parsed, "image_ratings": ratings,
                      "calls": judge["model"].calls[before:]}
            atomic_json(score_path, result)
        correct = claim["label"].casefold() == prediction["verdict"].strip().casefold()
        scores.append({"claim_id": claim_id, "configuration": key, "score_file": score_path.name,
                       "semantic_recall": result["recall"],
                       "paper_rule_recall": result["paper_rule_recall"],
                       "notebook_rule_verdict": int(correct and result["recall"] > 0.4),
                       "paper_rule_verdict": int(correct and result["paper_rule_recall"] >= 0.3)})
        print(f"Scored development ordinal {index}, configuration {key}", flush=True)
    atomic_json(args.output / "scores.json", scores)
    aggregates = {}
    for key in sorted({row["configuration"] for row in scores}):
        selected = [row for row in scores if row["configuration"] == key]
        aggregates[key] = {"claims": len(selected),
            **{name: statistics.mean(row[name] for row in selected)
               for name in ("semantic_recall", "paper_rule_recall",
                            "notebook_rule_verdict", "paper_rule_verdict")}}
    atomic_json(args.output / "summary.json", {
        "mode": args.mode, "cases": len(scores), "aggregates": aggregates,
        "generation_calls_this_process": judge["model"].calls,
        "test_annotations_accessed": False,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("preflight", "pilot", "full"), required=True)
    parser.add_argument("--organizer", type=lambda p: Path(p).resolve(), required=True)
    parser.add_argument("--model-dir", type=lambda p: Path(p).resolve())
    parser.add_argument("--output", type=lambda p: Path(p).resolve(), required=True)
    parser.add_argument("--gpu-gib", type=int, default=19)
    parser.add_argument("--cpu-gib", type=int, default=37)
    args = parser.parse_args()
    experiments = Path(__file__).resolve().parents[1]
    root = experiments / "outputs/external_confirmation/averimatec"
    claims, jobs, references, images, hashes = prepare(experiments, root)
    pilot = choose_pilot(claims, jobs)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.mode == "preflight":
        report = {"claims": len(claims), "outputs": len(claims) * len(jobs),
                  "reference_images": len(references), "selected_image_files": len(images),
                  "reference_questions": sum(len(row["questions"]) for row in claims),
                  "distinct_prediction_sets": len({digest([index, rows[index]["evidence"]])
                       for index in range(len(claims)) for rows in jobs.values()}),
                  "pilot": pilot, "frozen_inputs": hashes, "test_annotations_accessed": False}
        atomic_json(args.output / "preflight.json", report)
        print(json.dumps(report, indent=2))
        return
    cases = pilot if args.mode == "pilot" else [
        (index, key) for index in range(len(claims)) for key in jobs]
    run_scoring(args, claims, jobs, references, images, hashes, cases)


if __name__ == "__main__":
    main()

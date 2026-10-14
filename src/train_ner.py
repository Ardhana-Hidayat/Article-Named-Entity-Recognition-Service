"""
Step 3: NER Model Training Pipeline
- Builds a baseline using pretrained spaCy en_core_web_sm
- Fine-tunes a new spaCy NER model on the training data  
- Evaluates both on validation and test sets
- Saves the best model artifact
"""
import json
import os
import sys
import time
import random
from collections import Counter, defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

import spacy
from spacy.training import Example
from spacy.util import minibatch, compounding
from spacy.tokens import DocBin
import numpy as np

SEED = 42
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "ner_finetuned")

# Supported labels
LABELS = ["PERSON", "ORG", "GPE", "DATE", "EVENT"]

# spaCy label mapping for baseline evaluation
SPACY_LABEL_MAP = {
    "PERSON": "PERSON",
    "ORG": "ORG",
    "GPE": "GPE",
    "LOC": "GPE",
    "DATE": "DATE",
    "TIME": "DATE",
    "EVENT": "EVENT",
}


def load_annotations(split_name):
    """Load annotations from JSONL file."""
    path = os.path.join(DATA_DIR, f"annotations_{split_name}.jsonl")
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def annotations_to_spacy_format(annotations):
    """Convert annotations to spaCy training format: (text, {"entities": [(start, end, label)]})."""
    training_data = []
    for ann in annotations:
        text = ann['combined_text']
        entities = []
        for ent in ann['entities']:
            if ent['label'] in LABELS:
                entities.append((ent['start'], ent['end'], ent['label']))
        
        # Remove overlapping entities (keep first one)
        entities = remove_overlaps(entities)
        training_data.append((text, {"entities": entities}))
    
    return training_data


def remove_overlaps(entities):
    """Remove overlapping entity spans, keeping the first one."""
    if not entities:
        return entities
    
    sorted_ents = sorted(entities, key=lambda x: (x[0], -(x[1] - x[0])))
    result = [sorted_ents[0]]
    
    for ent in sorted_ents[1:]:
        prev = result[-1]
        if ent[0] >= prev[1]:  # No overlap
            result.append(ent)
    
    return result


def evaluate_model(nlp, data, label_map=None):
    """
    Evaluate NER model using exact-span matching.
    Returns overall and per-label P/R/F1.
    """
    tp = defaultdict(int)  # True positives per label
    fp = defaultdict(int)  # False positives per label
    fn = defaultdict(int)  # False negatives per label
    
    for ann in data:
        text = ann['combined_text']
        doc = nlp(text)
        
        # Get predicted entities
        pred_ents = set()
        for ent in doc.ents:
            mapped = label_map.get(ent.label_, ent.label_) if label_map else ent.label_
            if mapped and mapped in LABELS:
                pred_ents.add((ent.start_char, ent.end_char, mapped))
        
        # Get gold entities
        gold_ents = set()
        for ent in ann['entities']:
            if ent['label'] in LABELS:
                gold_ents.add((ent['start'], ent['end'], ent['label']))
        
        # Compute TP, FP, FN
        for ent in pred_ents:
            if ent in gold_ents:
                tp[ent[2]] += 1
            else:
                fp[ent[2]] += 1
        
        for ent in gold_ents:
            if ent not in pred_ents:
                fn[ent[2]] += 1
    
    # Calculate per-label metrics
    results = {}
    all_tp = sum(tp.values())
    all_fp = sum(fp.values())
    all_fn = sum(fn.values())
    
    for label in LABELS:
        p = tp[label] / (tp[label] + fp[label]) if (tp[label] + fp[label]) > 0 else 0.0
        r = tp[label] / (tp[label] + fn[label]) if (tp[label] + fn[label]) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        results[label] = {
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f1, 4),
            "tp": tp[label],
            "fp": fp[label],
            "fn": fn[label],
            "support": tp[label] + fn[label],
        }
    
    # Overall micro-averaged metrics
    overall_p = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0.0
    overall_r = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0.0
    overall_f1 = 2 * overall_p * overall_r / (overall_p + overall_r) if (overall_p + overall_r) > 0 else 0.0
    
    results["OVERALL"] = {
        "precision": round(overall_p, 4),
        "recall": round(overall_r, 4),
        "f1": round(overall_f1, 4),
        "tp": all_tp,
        "fp": all_fp,
        "fn": all_fn,
        "support": all_tp + all_fn,
    }
    
    return results


def print_eval_results(results, model_name):
    """Print evaluation results as a formatted table."""
    print(f"\n{'='*70}")
    print(f"Evaluation Results: {model_name}")
    print(f"{'='*70}")
    print(f"{'Label':<12} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print(f"{'-'*52}")
    
    for label in LABELS:
        if label in results:
            r = results[label]
            print(f"{label:<12} {r['precision']:>10.4f} {r['recall']:>10.4f} {r['f1']:>10.4f} {r['support']:>10}")
    
    print(f"{'-'*52}")
    r = results["OVERALL"]
    print(f"{'OVERALL':<12} {r['precision']:>10.4f} {r['recall']:>10.4f} {r['f1']:>10.4f} {r['support']:>10}")
    print()


def error_analysis(nlp, test_data, label_map=None, n_errors=5):
    """Find and describe notable errors from test set."""
    errors = []
    
    for ann in test_data:
        text = ann['combined_text']
        doc = nlp(text)
        
        pred_ents = set()
        pred_details = {}
        for ent in doc.ents:
            mapped = label_map.get(ent.label_, ent.label_) if label_map else ent.label_
            if mapped and mapped in LABELS:
                key = (ent.start_char, ent.end_char, mapped)
                pred_ents.add(key)
                pred_details[key] = ent.text
        
        gold_ents = set()
        gold_details = {}
        for ent in ann['entities']:
            if ent['label'] in LABELS:
                key = (ent['start'], ent['end'], ent['label'])
                gold_ents.add(key)
                gold_details[key] = ent['text']
        
        # False positives
        for ent in pred_ents - gold_ents:
            context_start = max(0, ent[0] - 30)
            context_end = min(len(text), ent[1] + 30)
            context = text[context_start:context_end]
            errors.append({
                "type": "FALSE_POSITIVE",
                "text": pred_details[ent],
                "predicted_label": ent[2],
                "gold_label": None,
                "context": f"...{context}...",
                "original_id": ann.get('original_id', 'unknown'),
            })
        
        # False negatives
        for ent in gold_ents - pred_ents:
            context_start = max(0, ent[0] - 30)
            context_end = min(len(text), ent[1] + 30)
            context = text[context_start:context_end]
            errors.append({
                "type": "FALSE_NEGATIVE",
                "text": gold_details[ent],
                "predicted_label": None,
                "gold_label": ent[2],
                "context": f"...{context}...",
                "original_id": ann.get('original_id', 'unknown'),
            })
    
    return errors[:n_errors * 3]  # Return more than needed for selection


def train_ner_model(train_data_spacy, val_data, n_epochs=30, batch_size=8, dropout=0.3, lr=0.001):
    """Fine-tune a spaCy NER model on training data."""
    # Create a blank English model
    nlp = spacy.blank("en")
    
    # Add NER pipe
    ner = nlp.add_pipe("ner")
    for label in LABELS:
        ner.add_label(label)
    
    # Set random seeds for reproducibility
    random.seed(SEED)
    np.random.seed(SEED)
    spacy.util.fix_random_seed(SEED)
    
    # Get training examples
    optimizer = nlp.begin_training()
    optimizer.learn_rate = lr
    
    best_f1 = 0.0
    best_epoch = 0
    patience = 5
    no_improve = 0
    
    print(f"\nTraining NER model for up to {n_epochs} epochs...")
    print(f"  Training examples: {len(train_data_spacy)}")
    print(f"  Batch size: {batch_size}, Dropout: {dropout}, LR: {lr}")
    
    start_time = time.time()
    
    for epoch in range(n_epochs):
        random.shuffle(train_data_spacy)
        losses = {}
        
        # Create Example objects for each batch
        batches = minibatch(train_data_spacy, size=compounding(4.0, batch_size, 1.001))
        
        for batch in batches:
            examples = []
            for text, annotations in batch:
                try:
                    doc = nlp.make_doc(text)
                    example = Example.from_dict(doc, annotations)
                    examples.append(example)
                except Exception:
                    continue
            
            if examples:
                nlp.update(examples, drop=dropout, losses=losses, sgd=optimizer)
        
        # Evaluate on validation set every epoch
        val_results = evaluate_model(nlp, val_data)
        val_f1 = val_results["OVERALL"]["f1"]
        
        print(f"  Epoch {epoch+1:3d} | Loss: {losses.get('ner', 0):.4f} | "
              f"Val F1: {val_f1:.4f} | Val P: {val_results['OVERALL']['precision']:.4f} | "
              f"Val R: {val_results['OVERALL']['recall']:.4f}")
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch + 1
            no_improve = 0
            # Save best model
            os.makedirs(MODEL_DIR, exist_ok=True)
            nlp.to_disk(MODEL_DIR)
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
                break
    
    duration = time.time() - start_time
    print(f"\nTraining complete in {duration:.1f}s")
    print(f"Best epoch: {best_epoch}, Best Val F1: {best_f1:.4f}")
    print(f"Model saved to {MODEL_DIR}")
    
    # Load best model
    best_nlp = spacy.load(MODEL_DIR)
    
    return best_nlp, {
        "duration_seconds": round(duration, 1),
        "best_epoch": best_epoch,
        "total_epochs": epoch + 1,
        "best_val_f1": best_f1,
        "learning_rate": lr,
        "dropout": dropout,
        "batch_size": batch_size,
        "n_training_examples": len(train_data_spacy),
    }


def main():
    print("="*60)
    print("STEP 3: NER MODEL TRAINING PIPELINE")
    print("="*60)
    
    # Load data
    print("\nLoading annotations...")
    train_data = load_annotations("train")
    val_data = load_annotations("val")
    test_data = load_annotations("test")
    
    print(f"Train: {len(train_data)} docs")
    print(f"Val:   {len(val_data)} docs")
    print(f"Test:  {len(test_data)} docs")
    
    # Count entities per split
    for name, data in [("Train", train_data), ("Val", val_data), ("Test", test_data)]:
        ent_counts = Counter(e['label'] for ann in data for e in ann['entities'])
        total_ents = sum(ent_counts.values())
        print(f"  {name}: {total_ents} entity spans - {dict(ent_counts)}")
    
    # ============================================================
    # BASELINE: Pretrained en_core_web_sm (no adaptation)
    # ============================================================
    print("\n" + "="*60)
    print("BASELINE MODEL: en_core_web_sm (pretrained, no adaptation)")
    print("="*60)
    
    baseline_nlp = spacy.load("en_core_web_sm")
    baseline_nlp.max_length = 200000
    
    print("\nEvaluating baseline on VALIDATION set...")
    baseline_val_results = evaluate_model(baseline_nlp, val_data, SPACY_LABEL_MAP)
    print_eval_results(baseline_val_results, "Baseline (Val)")
    
    print("Evaluating baseline on TEST set...")
    baseline_test_results = evaluate_model(baseline_nlp, test_data, SPACY_LABEL_MAP)
    print_eval_results(baseline_test_results, "Baseline (Test)")
    
    # ============================================================
    # TRAINED MODEL: Fine-tuned on training data
    # ============================================================
    print("\n" + "="*60)
    print("TRAINING: Fine-tuning NER model on training data")
    print("="*60)
    
    # Convert training data to spaCy format
    train_spacy = annotations_to_spacy_format(train_data)
    print(f"\nConverted {len(train_spacy)} training examples to spaCy format")
    
    # Filter out examples that are too long (to avoid memory issues)
    MAX_TEXT_LEN = 50000
    train_spacy_filtered = [(t, a) for t, a in train_spacy if len(t) <= MAX_TEXT_LEN]
    print(f"After filtering (max {MAX_TEXT_LEN} chars): {len(train_spacy_filtered)} examples")
    
    # Train the model
    trained_nlp, train_meta = train_ner_model(
        train_spacy_filtered, val_data,
        n_epochs=30, batch_size=8, dropout=0.35, lr=0.001
    )
    
    # ============================================================
    # EVALUATION: Trained model on val and test
    # ============================================================
    print("\n" + "="*60)
    print("EVALUATION: Trained/Adapted Model")
    print("="*60)
    
    print("\nEvaluating trained model on VALIDATION set...")
    trained_val_results = evaluate_model(trained_nlp, val_data)
    print_eval_results(trained_val_results, "Trained (Val)")
    
    print("Evaluating trained model on TEST set...")
    trained_test_results = evaluate_model(trained_nlp, test_data)
    print_eval_results(trained_test_results, "Trained (Test)")
    
    # ============================================================
    # COMPARISON TABLE
    # ============================================================
    print("\n" + "="*60)
    print("COMPARISON: Baseline vs Trained Model")
    print("="*60)
    print(f"\n{'Metric':<20} {'Baseline (Test)':>16} {'Trained (Test)':>16} {'Delta':>10}")
    print(f"{'-'*62}")
    for label in LABELS + ["OVERALL"]:
        b = baseline_test_results.get(label, {})
        t = trained_test_results.get(label, {})
        if b and t:
            b_f1 = b.get('f1', 0)
            t_f1 = t.get('f1', 0)
            delta = t_f1 - b_f1
            sign = "+" if delta >= 0 else ""
            print(f"{label:<20} {b_f1:>16.4f} {t_f1:>16.4f} {sign}{delta:>9.4f}")
    
    # ============================================================
    # ERROR ANALYSIS
    # ============================================================
    print("\n" + "="*60)
    print("ERROR ANALYSIS (on test set)")
    print("="*60)
    
    errors = error_analysis(trained_nlp, test_data)
    
    # Select 3 diverse errors
    fp_errors = [e for e in errors if e['type'] == 'FALSE_POSITIVE']
    fn_errors = [e for e in errors if e['type'] == 'FALSE_NEGATIVE']
    
    selected_errors = []
    if fn_errors:
        selected_errors.append(fn_errors[0])
    if fp_errors:
        selected_errors.append(fp_errors[0])
    if len(fn_errors) > 1:
        selected_errors.append(fn_errors[1])
    elif len(fp_errors) > 1:
        selected_errors.append(fp_errors[1])
    # Fill up to 3 if needed
    remaining = [e for e in errors if e not in selected_errors]
    while len(selected_errors) < 3 and remaining:
        selected_errors.append(remaining.pop(0))
    
    for i, err in enumerate(selected_errors[:5], 1):
        print(f"\n--- Error {i}: {err['type']} ---")
        print(f"  Entity text: \"{err['text']}\"")
        print(f"  Predicted label: {err['predicted_label']}")
        print(f"  Gold label: {err['gold_label']}")
        print(f"  Context: {err['context'][:150]}")
        if err['type'] == 'FALSE_POSITIVE':
            print(f"  Analysis: Model incorrectly predicted this span as {err['predicted_label']}.")
            print(f"  Possible cause: The model may be over-generalizing from training patterns.")
        else:
            print(f"  Analysis: Model missed this {err['gold_label']} entity.")
            print(f"  Possible cause: Insufficient training examples or unfamiliar entity pattern.")
    
    # ============================================================
    # Save all evaluation results
    # ============================================================
    eval_results = {
        "seed": SEED,
        "matching_scheme": "exact-span matching (start, end, label must all match)",
        "test_set_isolation": "Test set was NOT used during annotation review, label mapping, hyperparameter tuning, or model selection. It was evaluated only once after the best model was selected on the validation set.",
        "training_config": train_meta,
        "baseline": {
            "model": "en_core_web_sm (spaCy 3.8.x pretrained, no adaptation)",
            "validation": baseline_val_results,
            "test": baseline_test_results,
        },
        "trained": {
            "model": "spaCy blank:en fine-tuned NER",
            "model_dir": MODEL_DIR,
            "validation": trained_val_results,
            "test": trained_test_results,
        },
        "error_analysis": [
            {
                "error_num": i+1,
                "type": err['type'],
                "text": err['text'],
                "predicted": err['predicted_label'],
                "gold": err['gold_label'],
                "context": err['context'][:200],
            }
            for i, err in enumerate(selected_errors[:5])
        ],
    }
    
    results_path = os.path.join(DATA_DIR, "evaluation_results.json")
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(eval_results, f, indent=2, ensure_ascii=False)
    print(f"\nEvaluation results saved to {results_path}")
    
    print("\n" + "="*60)
    print("STEP 3 COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()

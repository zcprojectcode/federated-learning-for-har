import torch
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import logging

def evaluate_local(local_models, selected, data_loader, num_classes, device, eval_time):
    """
    Evaluate the HAR performance local model for relevant clients

    Args:
        local_models: local model for each client
        selected: list of the selected clients in global aggregation round
        data_loader: training and test data for each client
        device: training platform
        eval_time: whether the metrics are for the global model ("before") or
        local model after local training ("after")
    
    Returns:
        accs: a list of the accuracies for each client
        metrics: a dictionary containing accuracy, f1 score, precision, recall
    """
    accs = []

    # Evaluate the performance of each local model on test data
    for i, model in enumerate(local_models):
        model.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            for x, y in data_loader.get_testing_idx(selected[i]):
                x = x.to(device)
                logits, _ = model(x)
                pred = logits.argmax(dim=1).cpu()
                all_preds.append(pred)
                all_labels.append(y)

        all_preds  = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()

        acc = 100.0 * (all_preds == all_labels).sum() / len(all_labels)
        logging.info(f"Client {selected[i]} accuracy: {acc}")

        accs.append(acc)
    
    if eval_time == "before":
        logging.info(f"Average accuracy {eval_time} training (all clients): {np.mean(accs)}")
    elif eval_time == "after":
        logging.info(f"Average accuracy {eval_time} training (selected clients): {np.mean(accs)}")

    # Per-class accuracy
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)), normalize="true")
    class_acc = cm.diagonal() / cm.sum(axis=1).clip(min=1)

    # Per-class precision, recall and F1 score
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, labels=list(range(num_classes)), zero_division=0
    )

    # Aggregate stats
    def _stats(arr):
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        }

    metrics = {
        "accuracy": acc,
        "f1_macro": float(f1.mean()),
        "f1_weighted": float(np.average(f1, weights=support)),
        "precision": _stats(precision),
        "recall": _stats(recall),
        "f1": _stats(f1),
        "per_class": {
            i: {
                "accuracy": float(class_acc[i]),
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i in range(len(class_acc))
        },
    }

    # Log results
    logging.info(f"Accuracy: {acc:.2f}%")
    logging.info(f"F1 (macro): {metrics['f1_macro']:.4f}")
    logging.info(f"F1 (weighted): {metrics['f1_weighted']:.4f}")

    for metric, label in [("precision", "Precision"), ("recall", "Recall"), ("f1", "F1")]:
        s = metrics[metric]
        logging.info(
            f"{label:<12} mean={s['mean']:.4f}  std={s['std']:.4f}"
            f"  min={s['min']:.4f}  max={s['max']:.4f}"
        )

    logging.info("Per-class breakdown:")
    logging.info(f"  {'Class':>6}  {'Acc':>7}  {'Prec':>7}  {'Rec':>7}  {'F1':>7}  {'N':>6}")
    for i, pc in metrics["per_class"].items():
        logging.info(
            f"  {i:>6}  {pc['accuracy']:>7.4f}  {pc['precision']:>7.4f}"
            f"  {pc['recall']:>7.4f}  {pc['f1']:>7.4f}  {pc['support']:>6}"
        )

    return accs, metrics
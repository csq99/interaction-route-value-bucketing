# DistilBERT random complete-route adapter ablation

This independent experiment compares adapter training on predictor-selected
complete routes against uniform random legal complete routes. The DistilBERT
backbone and the already-trained Early-CLS predictor are frozen. Both branches
start from the same random adapter initialization and use the same CE plus
executed-layer CLS hidden-state KD objective. Accuracy is reported for the
predictor's actual routes and for every legal route at the target budget.

import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras import layers, models
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
import json
import datetime

# VERİSETİ HAZIRLIK
def prepare_datasets(data_dir, img_size, batch_size, seed=123):
    train_ds = image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=seed,
        image_size=img_size,
        batch_size=batch_size
    )
    val_ds = image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=seed,
        image_size=img_size,
        batch_size=batch_size
    )
    return train_ds.prefetch(tf.data.AUTOTUNE), val_ds.prefetch(tf.data.AUTOTUNE), train_ds.class_names

# MODEL OLUŞTURMA
def build_model(model_name, input_size, num_classes, dropout_rate, dense_units):
    base_model_map = {
        "mobilenet": tf.keras.applications.MobileNetV2,
        "resnet": tf.keras.applications.ResNet50
    }
    if model_name.lower() not in base_model_map:
        raise ValueError("Desteklenmeyen model: {}".format(model_name))

    base_model = base_model_map[model_name.lower()] (
        input_shape=(*input_size, 3), include_top=False, weights="imagenet"
    )
    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(dense_units, activation="relu"),
        layers.Dropout(dropout_rate),
        layers.Dense(num_classes, activation="softmax")
    ])
    return model

# EĞİTİM VE DEĞERLENDİRME
def train_and_evaluate_model(model_name, train_ds, val_ds, input_size, num_classes,
                             dropout_rate, dense_units, use_augmentation, epochs,
                             class_names, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    if use_augmentation:
        aug_layer = tf.keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
        ])
        train_ds = train_ds.map(lambda x, y: (aug_layer(x), y))

    model = build_model(model_name, input_size, num_classes, dropout_rate, dense_units)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    summary_path = os.path.join(output_dir, "model_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        model.summary(print_fn=lambda x: f.write(x + "\n"))

    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)

    df_log = pd.DataFrame({
        "epoch": list(range(1, epochs + 1)),
        "train_accuracy": history.history["accuracy"],
        "val_accuracy": history.history["val_accuracy"],
        "train_loss": history.history["loss"],
        "val_loss": history.history["val_loss"]
    })
    df_log.to_csv(os.path.join(output_dir, "training_log.csv"), index=False)

    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump({
            "model_name": model_name,
            "dropout_rate": dropout_rate,
            "dense_units": dense_units,
            "use_augmentation": use_augmentation,
            "epochs": epochs,
            "input_size": input_size,
            "num_classes": num_classes
        }, f, indent=4)

    plot_metrics(history, output_dir)

    y_true, y_pred, y_probs = [], [], []
    for x_batch, y_batch in val_ds:
        preds = model.predict(x_batch)
        y_true.extend(y_batch.numpy())
        y_pred.extend(np.argmax(preds, axis=1))
        y_probs.extend(preds)

    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    pd.DataFrame(report).transpose().to_csv(os.path.join(output_dir, "classification_report.csv"))

    cm = confusion_matrix(y_true, y_pred, normalize='true')
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, xticklabels=class_names, yticklabels=class_names, fmt=".2f", cmap="Blues")
    plt.title("Normalized Confusion Matrix")
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    plt.close()

    pred_df = pd.DataFrame({
        "True Label": [class_names[i] for i in y_true],
        "Predicted Label": [class_names[i] for i in y_pred],
        "Correct": [yt == yp for yt, yp in zip(y_true, y_pred)]
    })
    pred_df.to_csv(os.path.join(output_dir, "predictions.csv"), index=False)

    misclassified_dir = os.path.join(output_dir, "misclassified_images")
    os.makedirs(misclassified_dir, exist_ok=True)
    i = 0
    for batch, (x_batch, y_batch) in enumerate(val_ds):
        preds = model.predict(x_batch)
        pred_labels = np.argmax(preds, axis=1)
        for j in range(len(y_batch)):
            if y_batch[j].numpy() != pred_labels[j]:
                img = x_batch[j].numpy()
                plt.imsave(os.path.join(misclassified_dir, f"img_{i}true{class_names[y_batch[j]]}pred{class_names[pred_labels[j]]}.png"), img.astype(np.uint8))
                i += 1

    if num_classes == 2:
        fpr, tpr, _ = roc_curve(y_true, [p[1] for p in y_probs])
        plt.figure()
        plt.plot(fpr, tpr, label=f"ROC curve (AUC = {auc(fpr, tpr):.2f})")
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        plt.savefig(os.path.join(output_dir, "ROC_curve.png"))
        plt.close()

        precision, recall, _ = precision_recall_curve(y_true, [p[1] for p in y_probs])
        plt.figure()
        plt.plot(recall, precision)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.savefig(os.path.join(output_dir, "precision_recall_curve.png"))
        plt.close()

    model.save_weights(os.path.join(output_dir, "model.weights.h5"))

    return model

#GRAFİK
def plot_metrics(history, output_dir):
    metrics = ["accuracy", "loss"]
    for m in metrics:
        plt.figure()
        plt.plot(history.history[m], label=f"Train {m.title()}")
        plt.plot(history.history[f"val_{m}"], label=f"Validation {m.title()}")
        plt.title(m.title())
        plt.xlabel("Epoch")
        plt.ylabel(m.title())
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f"{m}.png"))
        plt.close()

# ANA AKIŞ
if __name__ == "__main__":
    from veriseti import path

    data_dir = path
    img_size = (224, 224)
    batch_size = 32
    num_classes = 4
    epochs = 5

    train_ds, val_ds, class_names = prepare_datasets(data_dir, img_size, batch_size)

    configs = [
        {"model_name": "mobilenet", "dropout_rate": 0.2, "dense_units": 64, "use_augmentation": True},
        {"model_name": "mobilenet", "dropout_rate": 0.2, "dense_units": 128, "use_augmentation": True},
        {"model_name": "mobilenet", "dropout_rate": 0.2, "dense_units": 256, "use_augmentation": True},
        {"model_name": "mobilenet", "dropout_rate": 0.4, "dense_units": 256, "use_augmentation": True},
        {"model_name": "mobilenet", "dropout_rate": 0.2, "dense_units": 64, "use_augmentation": False},
        {"model_name": "mobilenet", "dropout_rate": 0.2, "dense_units": 128, "use_augmentation": False},
        {"model_name": "mobilenet", "dropout_rate": 0.4, "dense_units": 256, "use_augmentation": False},
        {"model_name": "mobilenet", "dropout_rate": 0.2, "dense_units": 256, "use_augmentation": False},
        
        {"model_name": "resnet", "dropout_rate": 0.2, "dense_units": 64, "use_augmentation": True},
        {"model_name": "resnet", "dropout_rate": 0.2, "dense_units": 128, "use_augmentation": True},
        {"model_name": "resnet", "dropout_rate": 0.2, "dense_units": 256, "use_augmentation": True},
        {"model_name": "resnet", "dropout_rate": 0.4, "dense_units": 256, "use_augmentation": True},
        {"model_name": "resnet", "dropout_rate": 0.2, "dense_units": 64, "use_augmentation": False},
        {"model_name": "resnet", "dropout_rate": 0.2, "dense_units": 128, "use_augmentation": False},
        {"model_name": "resnet", "dropout_rate": 0.4, "dense_units": 256, "use_augmentation": False},
        {"model_name": "resnet", "dropout_rate": 0.2, "dense_units": 256, "use_augmentation": False},
        
    ]

    for i, cfg in enumerate(configs, 1):
        out_dir = f"results/{cfg['model_name'].upper()}Test{i}"
        train_and_evaluate_model(
            model_name=cfg["model_name"],
            train_ds=train_ds,
            val_ds=val_ds,
            input_size=img_size,
            num_classes=num_classes,
            dropout_rate=cfg["dropout_rate"],
            dense_units=cfg["dense_units"],
            use_augmentation=cfg["use_augmentation"],
            epochs=epochs,
            class_names=class_names,
            output_dir=out_dir
        )
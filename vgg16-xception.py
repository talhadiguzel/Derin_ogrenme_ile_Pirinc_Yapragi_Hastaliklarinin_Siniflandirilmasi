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

#VERİSETİ HAZIRLIK
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
        "vgg16": tf.keras.applications.VGG16,
        "xception": tf.keras.applications.Xception
    }

    if model_name.lower() not in base_model_map:
        raise ValueError(f"Desteklenmeyen model: {model_name}")

    base_model = base_model_map[model_name.lower()](
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
    for x_batch, y_batch in val_ds:
        preds = model.predict(x_batch)
        pred_labels = np.argmax(preds, axis=1)
        for j in range(len(y_batch)):
            if y_batch[j].numpy() != pred_labels[j]:
                img = x_batch[j].numpy()
                plt.imsave(os.path.join(misclassified_dir, f"img_{i}_true_{class_names[y_batch[j]]}_pred_{class_names[pred_labels[j]]}.png"), img.astype(np.uint8))
                i += 1

    model.save_weights(os.path.join(output_dir, "model.weights.h5"))
    return model

# GRAFİK
def plot_metrics(history, output_dir):
    for metric in ["accuracy", "loss"]:
        plt.figure()
        plt.plot(history.history[metric], label=f"Train {metric.title()}")
        plt.plot(history.history[f"val_{metric}"], label=f"Validation {metric.title()}")
        plt.title(metric.title())
        plt.xlabel("Epoch")
        plt.ylabel(metric.title())
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f"{metric}.png"))
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
        {"model_name": "vgg16", "dropout_rate": 0.2, "dense_units": 64, "use_augmentation": True},
        {"model_name": "vgg16", "dropout_rate": 0.2, "dense_units": 128, "use_augmentation": True},
        {"model_name": "vgg16", "dropout_rate": 0.2, "dense_units": 256, "use_augmentation": True},
        {"model_name": "vgg16", "dropout_rate": 0.4, "dense_units": 256, "use_augmentation": True},
        {"model_name": "vgg16", "dropout_rate": 0.2, "dense_units": 64, "use_augmentation": False},
        {"model_name": "vgg16", "dropout_rate": 0.2, "dense_units": 128, "use_augmentation": False},
        {"model_name": "vgg16", "dropout_rate": 0.4, "dense_units": 256, "use_augmentation": False},
        {"model_name": "vgg16", "dropout_rate": 0.2, "dense_units": 256, "use_augmentation": False},

        {"model_name": "xception", "dropout_rate": 0.2, "dense_units": 64, "use_augmentation": True},
        {"model_name": "xception", "dropout_rate": 0.2, "dense_units": 128, "use_augmentation": True},
        {"model_name": "xception", "dropout_rate": 0.2, "dense_units": 256, "use_augmentation": True},
        {"model_name": "xception", "dropout_rate": 0.4, "dense_units": 256, "use_augmentation": True},
        {"model_name": "xception", "dropout_rate": 0.2, "dense_units": 64, "use_augmentation": False},
        {"model_name": "xception", "dropout_rate": 0.2, "dense_units": 128, "use_augmentation": False},
        {"model_name": "xception", "dropout_rate": 0.4, "dense_units": 256, "use_augmentation": False},
        {"model_name": "xception", "dropout_rate": 0.2, "dense_units": 256, "use_augmentation": False},
    ]

    for i, cfg in enumerate(configs, 1):
        out_dir = f"results/{cfg['model_name'].upper()}_Test_{i}"
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

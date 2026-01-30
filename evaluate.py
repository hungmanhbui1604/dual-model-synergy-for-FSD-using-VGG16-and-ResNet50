import argparse
import os
import yaml
import torch
import numpy as np
from tqdm import tqdm

from models.binary_models import get_model
from data_utils.data_loaders import FinPADDataset, TransformedDataset
from data_utils.transforms import get_transforms
from torch.utils.data import DataLoader
from train_utils.metrics import compute_metrics, find_optimal_threshold


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def evaluate(model, data_loader, device, threshold=None):
    model.eval()
    all_labels = []
    all_probabilities = []

    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc="Evaluating"):
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            probabilities = torch.sigmoid(logits)

            all_labels.append(labels.cpu().numpy())
            all_probabilities.append(probabilities.cpu().numpy())

    all_labels = np.concatenate(all_labels)
    all_probabilities = np.concatenate(all_probabilities)

    if threshold is None:
        threshold, apcer, bpcer, ace, accuracy = find_optimal_threshold(
            all_labels, all_probabilities, based_on="ace"
        )
    else:
        predictions = (all_probabilities >= threshold).astype(int)
        apcer, bpcer, ace, accuracy = compute_metrics(all_labels, predictions)
        apcer *= 100.0
        bpcer *= 100.0
        ace *= 100.0
        accuracy *= 100.0

    return {
        'threshold': threshold,
        'apcer': apcer,
        'bpcer': bpcer,
        'ace': ace,
        'accuracy': accuracy
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained FSD model')
    parser.add_argument('--config', type=str, default='configs/eval_config.yaml',
                        help='Path to evaluation config file')
    args = parser.parse_args()

    # Load configuration
    print(f"Loading configuration from {args.config}")
    config = load_config(args.config)

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    model_name = config.get('MODEL_NAME', 'dual_model')
    model_save_path = config['MODEL_LOAD_PATH']
    print(f"Loading {model_name} from {model_save_path}")

    checkpoint = torch.load(model_save_path, map_location=device)
    model = get_model(model_name, out_dim=1)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    # Load data
    test_sensor_path = config['TEST_SENSOR_PATH']
    multiclass = config.get('MULTICLASS', False)
    transform_type = config.get('TRANSFORM_TYPE', 'dual_model_transform')
    batch_size = config.get('BATCH_SIZE', 32)
    num_workers = config.get('NUM_WORKERS', 0)

    print(f"Loading test data from {test_sensor_path}")
    test_dataset = FinPADDataset(test_sensor_path, train=False, multiclass=multiclass)
    transforms = get_transforms(transform_type)
    test_set = TransformedDataset(test_dataset, transforms['Test'])

    use_pin_memory = torch.cuda.is_available()
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory
    )

    print(f"Test dataset size: {len(test_dataset)}")
    print(f"Number of test batches: {len(test_loader)}")

    # Evaluate
    threshold = config.get('THRESHOLD', None)
    results = evaluate(model, test_loader, device, threshold=threshold)

    # Print results
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    if threshold is None:
        print(f"Optimal Threshold: {results['threshold']:.6f}")
    else:
        print(f"Threshold: {results['threshold']:.6f}")
    print(f"APCER: {results['apcer']:.2f}%")
    print(f"BPCER: {results['bpcer']:.2f}%")
    print(f"ACE: {results['ace']:.2f}%")
    print(f"Accuracy: {results['accuracy']:.2f}%")
    print("=" * 50)

    # Save results if path provided
    results_save_path = config.get('RESULTS_SAVE_PATH')
    if results_save_path:
        os.makedirs(os.path.dirname(results_save_path), exist_ok=True)
        results_to_save = {
            'model_path': model_save_path,
            'test_data_path': test_sensor_path,
            **results
        }
        torch.save(results_to_save, results_save_path)
        print(f"Results saved to {results_save_path}")

        # Save to txt file
        txt_save_path = results_save_path.replace('.pth', '.txt')
        with open(txt_save_path, 'w') as f:
            f.write("=" * 50 + "\n")
            f.write("EVALUATION RESULTS\n")
            f.write("=" * 50 + "\n")
            f.write(f"Model Path: {model_save_path}\n")
            f.write(f"Test Data Path: {test_sensor_path}\n")
            f.write(f"Test Dataset Size: {len(test_dataset)}\n")
            f.write(f"-" * 50 + "\n")
            if threshold is None:
                f.write(f"Optimal Threshold: {results['threshold']:.6f}\n")
            else:
                f.write(f"Threshold: {results['threshold']:.6f}\n")
            f.write(f"APCER: {results['apcer']:.2f}%\n")
            f.write(f"BPCER: {results['bpcer']:.2f}%\n")
            f.write(f"ACE: {results['ace']:.2f}%\n")
            f.write(f"Accuracy: {results['accuracy']:.2f}%\n")
            f.write("=" * 50 + "\n")
        print(f"Results saved to {txt_save_path}")


if __name__ == '__main__':
    main()

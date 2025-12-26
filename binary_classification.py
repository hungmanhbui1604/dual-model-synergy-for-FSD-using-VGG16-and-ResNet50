import torch
import argparse
import yaml
from data_utils.data_loaders import get_data_loaders
from data_utils.transforms import get_transforms
from models.binary_models import get_model
from train_utils.trainer import BinaryTrainer

def parse_args():
    parser = argparse.ArgumentParser(description='Binary Classification')
    parser.add_argument('-c', '--config', type=str, required=True, help='Path to configuration file')
    return parser.parse_args()

def get_training_components(model, config):
    if config['CRITERION_TYPE'] == 'bce_with_logits':
        criterion = torch.nn.BCEWithLogitsLoss()
    if config['OPTIMIZER_TYPE'] == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=config['LEARNING_RATE'], weight_decay=config['WEIGHT_DECAY'])
    if config['SCHEDULER_TYPE'] == 'cosine_annealing':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['NUM_EPOCHS'], eta_min=1e-6)
    return criterion, optimizer, scheduler

def main():
    args = parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # data loaders
    train_loader, val_loader, test_loader, _ = get_data_loaders(
        train_sensor_path=config['TRAIN_SENSOR_PATH'],
        test_sensor_path=config['TEST_SENSOR_PATH'],
        multiclass=False,
        transform=get_transforms(config['TRANSFORM_TYPE']),
        batch_size=config['BATCH_SIZE'],
        num_workers=config['NUM_WORKERS'],
        val_split=config['VAL_SPLIT'],
        seed=config['SEED']
    )

    # model
    model = get_model(
        model_name=config['MODEL_NAME'],
        out_dim=1
    )
    model.to(device)

    # criterion, optimizer, scheduler
    criterion, optimizer, scheduler = get_training_components(model, config)

    # trainer
    trainer = BinaryTrainer()
    trainer.run(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        num_epochs=config['NUM_EPOCHS'],
        model_save_path=config['MODEL_SAVE_PATH'],
        result_save_path=config['RESULTS_SAVE_PATH'],
        train_threshold=0.5,
        based_on='ace'
    )

if __name__ == '__main__':
    main()
from data_loader.data_generator import DataLoader
from model.livenessnet import LivenessNet
from trainer.trainer import Trainer

def main():
    dataloader = DataLoader(config="config.json")
    model = LivenessNet().build()
    trainer = Trainer(dataloader, model, config="config.json")
    h = trainer.train()

if __name__ == "__main__":
    main()
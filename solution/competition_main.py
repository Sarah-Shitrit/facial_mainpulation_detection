## competition_main.py

"""This is the script we will use to evaluate your model. Don't change it."""
from torch import nn
from trainer import Trainer
from competition_model import my_competition_model
from utils import load_dataset, get_nof_params


def main() -> tuple[float, int]:
        """Load model and test dataset, evaluate model on dataset and report the
        number of model parameters.

        Returns:
            test_accuracy, nof_parameters: test accuracy (float) and the number
            of model parameters (int).
        """
        test_dataset = load_dataset('fakes_dataset', 'test')

        model = my_competition_model()
        trainer = Trainer(model=model, optimizer=None,
                          criterion=nn.CrossEntropyLoss(), batch_size=128,
                          train_dataset=test_dataset,
                          validation_dataset=test_dataset,
                          test_dataset=test_dataset)
        _, test_accuracy = trainer.test()

        nof_parameters = get_nof_params(model)

     return test_accuracy, nof_parameters


if __name__ == "__main__":
    accuracy, nof_params = main()
    print(f"Your competition model test accuracy: {accuracy}")
    print(f"Your competition model number of parameters: {nof_params}")

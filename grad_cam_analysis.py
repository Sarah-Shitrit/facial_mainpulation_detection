"""Show network train graphs and analyze training results."""
import os
import argparse

import torch
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

from common import FIGURES_DIR
from utils import load_dataset, load_model


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Arguments
def parse_args():
    """Parse script arguments.

    Returns:
        Namespace with model name, checkpoint path and dataset name.
    """
    parser = argparse.ArgumentParser(description='Analyze network performance.')
    parser.add_argument('--model', '-m',
                        default='XceptionBased', type=str,
                        help='Model name: SimpleNet or XceptionBased.')
    parser.add_argument('--checkpoint_path', '-cpp',
                        default='checkpoints/XceptionBased.pt', type=str,
                        help='Path to model checkpoint.')
    parser.add_argument('--dataset', '-d',
                        default='fakes_dataset', type=str,
                        help='Dataset: fakes_dataset or synthetic_dataset.')

    return parser.parse_args()


def get_grad_cam_visualization(test_dataset: torch.utils.data.Dataset,
                               model: torch.nn.Module) -> tuple([np.ndarray,
                                                                torch.tensor]):
    """Return a tuple with the GradCAM visualization and true class label.

    Args:
        test_dataset: test dataset to choose a sample from.
        model: the model we want to understand.

    Returns:
        (visualization, true_label): a tuple containing the visualization of
        the conv3's response on one of the sample (256x256x3 np.ndarray) and
        the true label of that sample (since it is an output of a DataLoader
        of batch size 1, it's a tensor of shape (1,)).
    """
    """INSERT YOUR CODE HERE, overrun return."""
    from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image

    data = DataLoader(test_dataset, batch_size=1, shuffle=True)
    for test_image, test_labels in data:
        sample_image, sample_label = test_image[0], test_labels[0]

    target_layers = [model.conv3]
    input_tensor = sample_image.unsqueeze(0)
    cam = GradCAM(model=model, target_layers=target_layers)

    target_category = None
    grayscale_cam = cam(input_tensor, target_category)
    grayscale_cam = grayscale_cam[0, :]
    img = sample_image.numpy()
    img = np.swapaxes(img, 0, 1)
    img = np.swapaxes(img, 1, 2)
    img = img / (np.amax(img))
    img = np.clip(img, 0, 1)

    visualization = show_cam_on_image(img, grayscale_cam, use_rgb=True)

    return visualization, sample_label  # instead of return np.random.rand(256, 256, 3), torch.randint(0, 2, (1,))

def main():
    """Create two GradCAM images, one of a real image and one for a fake
    image for the model and dataset it receives as script arguments."""
    args = parse_args()
    test_dataset = load_dataset(dataset_name=args.dataset, dataset_part='test')

    model_name = args.model
    model = load_model(model_name)
    model.load_state_dict(torch.load(args.checkpoint_path)['model'])

    model.eval()
    seen_labels = []
    while len(set(seen_labels)) != 2:
        visualization, true_label = get_grad_cam_visualization(test_dataset,
                                                               model)
        grad_cam_figure = plt.figure()
        plt.imshow(visualization)
        title = 'Fake Image' if true_label == 1 else 'Real Image'
        plt.title(title)
        seen_labels.append(true_label.item())
        grad_cam_figure.savefig(
            os.path.join(FIGURES_DIR,
                         f'{args.dataset}_{args.model}_'
                         f'{title.replace(" ", "_")}_grad_cam.png'))


if __name__ == "__main__":
    main()

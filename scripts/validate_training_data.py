from scripts import validate_dataset, test_dataloader
from src.datasets.sen12ms import SEN12MSDataset
from src.utils import paths
from src.preprocessing import transform

if __name__ == "__main__":
    training_dataset = SEN12MSDataset(
        manifest_path=paths.TRAIN_LABELS_PATH,
        transform=transform.transform,
        sensors={"s2":[1,2,3,4,5,6,7,8]},
        compute_ndvi=True
    )

    validate_dataset.validate_dataset(dataset=training_dataset)
    test_dataloader.test_dataloader(
        dataset=training_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=0
    )
    
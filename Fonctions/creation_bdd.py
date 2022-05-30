import fiftyone as fo
import fiftyone.zoo as foz


dataset = foz.load_zoo_dataset(
    "open-images-v6", split="validation", classes=['Handgun'], label_types=["classifications"],
    max_samples=10,
    shuffle=True,
)
session = fo.launch_app(dataset)
session.wait()

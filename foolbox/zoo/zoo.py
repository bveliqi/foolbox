from .git_cloner import clone
from .model_loader import ModelLoader


def get_model(url: str):
    repo_path = clone(url)
    loader = ModelLoader.get()
    model = loader.load(repo_path)

    return model
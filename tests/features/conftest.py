import pytest
from pygenn import GeNNModel

@pytest.fixture
def make_model():
    created_models = []
    def _make_model(precision, name, backend):
        model = GeNNModel(precision, name, backend=backend)
        created_models.append(model)
        return model

    yield _make_model

    # Unload models
    for model in created_models:
        model.unload()

import pytest

from pygenn import GeNNModel

from pygenn.genn_model import backend_modules

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

@pytest.fixture
def backend(request):
    return request.param

@pytest.fixture
def backend_simt(request):
    return request.param

@pytest.fixture
def batch_size(request):
    return request.param

def pytest_generate_tests(metafunc):
    backend_simt_param = ("backend_simt" in metafunc.fixturenames)
    backend_param = ("backend" in metafunc.fixturenames)
    batch_size_param = ("batch_size" in metafunc.fixturenames)
    if backend_simt_param:
        assert not batch_size_param
        metafunc.parametrize("backend_simt", 
                             [b for b in backend_modules.keys() if b != "single_threaded_cpu"],
                             indirect=True)
                             
    if backend_param and batch_size_param:
        params = []
        for b in backend_modules.keys():
            params.append((b, 1))
            
            if b != "single_threaded_cpu":
                params.append((b, 5))
    
        metafunc.parametrize("backend, batch_size", params, indirect=True)
    elif backend_param:
        metafunc.parametrize("backend", backend_modules.keys(), indirect=True)

import pytest
import tensorflow as tf


@pytest.fixture(scope="session")
def disable_traceback_filtering(request):
    pass
    # tf.debugging.disable_traceback_filtering()

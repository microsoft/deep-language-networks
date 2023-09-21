from engine.scorer import prepare_prompts_scoring_args, prepare_inputs_scoring_args
import numpy as np


def test_prepare_args():
    class DummyLayer():
        def instantiate_template(self, inputs, prompt):
            return [prompt + "--" + input for input in inputs]

    result = prepare_prompts_scoring_args(DummyLayer(), np.array(["a", "b"]), np.array(["1", "2"]), np.array(["px", "py", "pz", "pk"]))
    result = np.array([c.context for c in result]).reshape(2, 1, 4)
    assert result[0, 0, 0] == 'px--a'
    assert result[0, 0, 1] == 'py--a'
    assert result[1, 0, 0] == 'px--b'

    result = prepare_inputs_scoring_args(DummyLayer(), np.array([["a1", "a2", "a3"], ["b1", "b2", "b3"]]), np.array(["1", "2"]), "px")
    result = np.array([c.context for c in result]).reshape(2, 1, 3)
    assert result[0, 0, 0] == 'px--a1'
    assert result[0, 0, 1] == 'px--a2'
    assert result[1, 0, 0] == 'px--b1'
    assert result[1, 0, 1] == 'px--b2'
    assert result[1, 0, 2] == 'px--b3'
    assert result[0, 0, 2] == 'px--a3'

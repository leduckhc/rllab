from rllab.core.parameterized import Parameterized


class VFunction(Parameterized):
    def get_vval(self, observations, **kwargs):
        raise NotImplementedError

    def get_vval_sym(self, obs_var, **kwargs):
        raise NotImplementedError

"""A wrapper swapped in for a module stands in for it where a library checks
the module's class."""

import mlx.core as mx
import mlx.nn as nn

from backcountry_runtime.swap import ReplayWrapper, install, uninstall


class ConvIn(nn.Conv2d):
    pass


class Decoder(nn.Module):
    """mflux's VAE block decoder: a conv picked by isinstance, then called
    through nn.Conv2d.__call__ (anything else is asked for its .conv)."""

    def __init__(self):
        super().__init__()
        self.conv_in = ConvIn(4, 4, 3, padding=1)

    def __call__(self, x):
        m = self.conv_in
        return nn.Conv2d.__call__(m if isinstance(m, nn.Conv2d) else m.conv, x)


class Observed(ReplayWrapper):
    def __call__(self, x):
        return self.wrapped(x)


def test_a_wrapper_answers_isinstance_as_the_module_it_wraps():
    # the FLUX.2 Klein job died at certification: the decoder found no .conv
    decoder = Decoder()
    x = mx.random.normal((1, 8, 8, 4))
    expected = decoder(x)
    occupant = install(decoder, "conv_in", Observed(decoder.conv_in))
    wrapper = decoder.conv_in
    assert isinstance(wrapper, ConvIn) and isinstance(wrapper, ReplayWrapper)
    assert type(wrapper) is Observed
    assert mx.array_equal(decoder(x), expected).item()
    uninstall(decoder, "conv_in", occupant)
    assert decoder.conv_in is occupant

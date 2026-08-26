"""Offline instruments with the public surfaces used by the real hardware."""

from collections.abc import Sequence

from qcodes.instrument import Instrument, InstrumentChannel
from qcodes.parameters import Parameter
from qcodes.validators import Enum, Numbers


class _OfflineInstrument(Instrument):
    def get_idn(self) -> dict[str, str | None]:
        return {
            "vendor": "QuantumDotControl",
            "model": self.__class__.__name__,
            "serial": "offline",
            "firmware": None,
        }


class DummyAgilent(_OfflineInstrument):
    """A minimal stand-in for an Agilent multimeter."""

    def __init__(
        self,
        name: str = "agilent",
        **kwargs,
    ):
        super().__init__(name, **kwargs)
        self.add_parameter(
            "volt",
            parameter_class=Parameter,
            initial_value=0.0,
            unit="V",
            vals=Numbers(),
            set_cmd=None,
            get_cmd=None,
        )
        self.add_parameter(
            "NPLC",
            parameter_class=Parameter,
            initial_value=1.0,
            vals=Numbers(0.001, 1000),
            get_cmd=None,
            set_cmd=None,
        )
        self.add_parameter(
            "range_auto",
            parameter_class=Parameter,
            initial_value="on",
            vals=Enum("on", "off"),
            get_cmd=None,
            set_cmd=None,
        )


class DummySpiDac(InstrumentChannel):
    """One SPI DAC module containing voltage-controlled DAC channels."""

    def __init__(
        self,
        parent: Instrument,
        name: str,
        dac_names: Sequence[str],
    ):
        super().__init__(parent, name)
        for dac_name in dac_names:
            self.add_submodule(dac_name, DummySpiChannel(self, dac_name))


class DummySpiChannel(InstrumentChannel):
    """One SPI DAC channel, exposed as ``dacN.voltage``."""

    def __init__(self, parent: InstrumentChannel, name: str):
        super().__init__(parent, name)
        self.add_parameter(
            "voltage",
            parameter_class=Parameter,
            initial_value=0.0,
            unit="V",
            vals=Numbers(-10, 10),
            get_cmd=None,
            set_cmd=None,
        )


class DummySpiRack(_OfflineInstrument):
    """A stateful stand-in for a Qblox SPI rack."""

    def __init__(self, name: str = "spi_rack", **kwargs):
        super().__init__(name, **kwargs)

    def add_spi_module(
        self,
        slot: int,
        module_type: str,
        name: str,
        dac_count: int = 16,
    ) -> DummySpiDac:
        """Add a module using the same initializer signature as the real rack."""
        del slot, module_type
        module = DummySpiDac(
            self,
            name,
            [f"dac{index}" for index in range(dac_count)],
        )
        self.add_submodule(name, module)
        return module


"""Offline instruments with the public surfaces used by the real hardware."""

from collections.abc import Sequence
import math
import random

from qcodes.instrument import Instrument, InstrumentChannel
from qcodes.parameters import Parameter
from qcodes.validators import Enum, Numbers


_MODELS: dict[str, "QuantumDotModel"] = {}


class QuantumDotModel:
    """Small, deterministic sensor model for offline autotuning experiments."""

    def __init__(self, seed: int = 0):
        self.voltages: dict[str, float] = {}
        self.rng = random.Random(seed)
        self.tripleDot_channels = {
            'left': {},
            'middle':{},
            'right':{}
        }
        self.sensor_channels = {
            "left": {
                "barriers": ("module1.dac8", "module1.dac7"),
                "plunger": "module2.dac2",
                "accumulation": ("module1.dac2",),
            },
            "right": {
                "barriers": ("module1.dac8", "module1.dac7"),
                "plunger": "module2.dac2",
                "accumulation": ("module1.dac2",),
            },
        }

    def configure_from_gates(self, gates: dict) -> None:
        """Map sensor behavior from the gate types in an all-in-one config."""
        sensor = self.sensor_channels["left"]
        barriers = []
        for details in gates.values():
            channel = details.get("channel", "").split(".", 1)[-1]
            channel = channel.rsplit(".", 1)[0]
            gate_type = details.get("type", "").lower()
            if "sensor plunger" in gate_type:
                sensor["plunger"] = channel
            elif "sensor barrier" in gate_type:
                barriers.append(channel)
            elif "sensor accumulation" in gate_type:
                sensor["accumulation"] = tuple(
                    set(sensor["accumulation"]) | {channel}
                )
        if len(barriers) >= 2:
            sensor["barriers"] = tuple(barriers[:2])

    def set_voltage(self, channel: str, value: float) -> None:
        self.voltages[channel] = value

    def measure(self, sensor: str = "left", noise: float = 0.0) -> float:
        """Return sensor voltage from barrier transmission and Coulomb peaks."""
        channels = self.sensor_channels.get(sensor, self.sensor_channels["left"])
        barrier_left = self.voltages.get(channels["barriers"][0], 0.0)
        barrier_right = self.voltages.get(channels["barriers"][1], 0.0)
        plunger = self.voltages.get(channels["plunger"], 0.0)
        accumulation = max(
            (self.voltages.get(channel, 0.0) for channel in channels["accumulation"]),
            default=0.0,
        )

        # Two Gaussian barriers give a tunable tunnel-coupling envelope.
        barrier_transmission = math.exp(-((barrier_left - 0.15) ** 2 + (barrier_right - 0.15) ** 2) / 0.08)
        peak_train = 0.08 + 0.92 * math.exp(
            5.0 * (math.cos(2.0 * math.pi * (plunger + 0.05) / 0.12) - 1.0)
        )
        accumulation_gate = 1.0 / (1.0 + math.exp(-10.0 * (accumulation - 0.1)))
        conductance = 1e-6 * barrier_transmission * accumulation_gate * peak_train
        voltage = conductance * 1e-3
        return voltage + self.rng.gauss(0.0, noise) if noise else voltage


def get_quantum_dot_model(model_id: str = "default", seed: int = 0) -> QuantumDotModel:
    """Return the shared model used by a dummy rack and its meters."""
    if model_id not in _MODELS:
        _MODELS[model_id] = QuantumDotModel(seed=seed)
    return _MODELS[model_id]


class _OfflineInstrument(Instrument):
    def get_idn(self) -> dict[str, str | None]:
        return {
            "vendor": "QuantumDotControl",
            "model": self.__class__.__name__,
            "serial": "offline",
            "firmware": None,
        }


class DummyAgilent(_OfflineInstrument):
    """A stateful stand-in for an Agilent 34410A/34401A multimeter."""

    def __init__(
        self,
        name: str = "agilent",
        model_id: str = "default",
        sensor: str = "left",
        noise: float = 0.0,
        **kwargs,
    ):
        super().__init__(name, **kwargs)
        model = get_quantum_dot_model(model_id)
        self.add_parameter(
            "volt",
            parameter_class=Parameter,
            get_cmd=lambda: model.measure(sensor, noise),
            initial_value=0.0,
            unit="V",
            vals=Numbers(),
            set_cmd=None,
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
        model = parent.root_instrument.model
        self.add_parameter(
            "voltage",
            parameter_class=Parameter,
            initial_value=0.0,
            unit="V",
            vals=Numbers(-10, 10),
            get_cmd=None,
            set_cmd=lambda value: model.set_voltage(
                f"{parent.short_name}.{name}", value
            ),
        )


class DummySpiRack(_OfflineInstrument):
    """A stateful stand-in for a Qblox SPI rack."""

    def __init__(self, name: str = "spi_rack", model_id: str = "default", seed: int = 0, **kwargs):
        super().__init__(name, **kwargs)
        self.model = get_quantum_dot_model(model_id, seed)

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

    def configure_from_gates(self, gates: dict) -> None:
        self.model.configure_from_gates(gates)
import qcodes
from qcodes import instrument_drivers
from qcodes.dataset import do0d, load_or_create_experiment
from qcodes.instrument import Instrument
from qcodes.instrument_drivers.stanford_research import SR830
from qcodes.validators import Numbers
from qcodes import Parameter

sr = SR830("lockin", "GPIB0::8::INSTR")


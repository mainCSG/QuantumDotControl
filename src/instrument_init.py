from qcodes.instrument import Instrument

def init_agilent(instrument, config):
    instrument.NPLC(config.get("NPLC", 1.0))
    instrument.range_auto(config.get("autorange", "on"))


def init_spi_rack(instrument, config):
    for name, module in config["modules"].items():
        instrument.add_spi_module(
            module["slot"],
            module["type"],
            name,
        )

def init_None(instrument, config):
    pass

INITIALIZERS = {'agilent': init_agilent, 'spi rack': init_spi_rack, 'None': init_None}
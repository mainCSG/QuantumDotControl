from qcodes.instrument import Instrument


def init_agilent(instrument: Instrument, *args):
    instrument.NPLC(1.0)
    instrument.range_auto('on')


def init_spi_rack(instrument: Instrument, logger):
    instrument.add_spi_module(8, 'D5a', 'module1')
    instrument.add_spi_module(7, 'D5a', 'module2')


INITIALIZERS = {
    'agilent_left': init_agilent,
    'agilent_right': init_agilent,
    'spi_rack': init_spi_rack,
}

MONITORS = {
    'agilent_left': ['volt'],
    'agilent_right': ['volt'],
}

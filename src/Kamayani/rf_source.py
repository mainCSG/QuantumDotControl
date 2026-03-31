import pyvisa

rm = pyvisa.ResourceManager()

# List all GPIB instruments
resources = rm.list_resources()
print("Available VISA resources:")
for res in resources:
    print(res)

# # Query a specific GPIB address (example: GPIB0::21::INSTR)
# # address = "GPIB0::19::INSTR"

# # try:
# #     instrument = rm.open_resource(address)
# #     idn = instrument.query("*IDN?")
# #     print(f"Instrument at {address}: {idn}")
# # except Exception as e:
# #     print(f"Could not connect to {address}: {e}")

from qcodes.instrument.visa import VisaInstrument

class AgilentE4432B(VisaInstrument):
    def __init__(self, name: str, address: str, **kwargs):
        super().__init__(name, address, **kwargs)
        self.connect_message()

    def get_idn(self):
        idn_str = self.ask('*IDN?')
        parts = idn_str.strip().split(',')
        return {
            'vendor': parts[0] if len(parts) > 0 else '',
            'model': parts[1] if len(parts) > 1 else '',
            'serial': parts[2] if len(parts) > 2 else '',
            'firmware': parts[3] if len(parts) > 3 else ''
        }

    def set_frequency(self, freq_hz: float):
        self.write(f'FREQ {freq_hz}HZ')

    def get_frequency(self) -> float:
        return float(self.ask('FREQ?'))

    def set_power(self, power_dbm: float):
        self.write(f'POW {power_dbm}DBM')

    def get_power(self) -> float:
        return float(self.ask('POW?'))

    def rf_on(self):
        self.write('OUTP ON')

    def rf_off(self):
        self.write('OUTP OFF')


# # Example usage
rf = AgilentE4432B('rf_gen', 'GPIB0::19::INSTR')
print(rf.get_idn())
freq = rf.get_frequency()  # 1 GHz
power = rf.get_power()      # -10 dBm
# # rf.rf_on()

print(f'Frequency: {freq} Hz')
print(f'Frequency: {power} dBm ')
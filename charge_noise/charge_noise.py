from charge_noise_tool import *

Palpatine = ChargeNoiseExtractor()

folder = "14-07-09_sweep2D_SDPvsV_SD_1"
data_name = "hallbar_V_SD_1_set_gates_SDP_set.dat"

# Load snapshot information
# into a dataframe (easier to process)
snapshot_json = pd.read_json(f"./data/{folder}/snapshot.json")
snapshot_df = pd.DataFrame(snapshot_json).replace(['#'],[''],regex=True)

# Load CSV into dataframe
data_csv = pd.read_csv(f"./data/{folder}/{data_name}", skiprows=[0,2], sep='\t')
data_df = pd.DataFrame(data_csv)

# Remove artifacts
data_df.columns = data_df.columns.str.replace('[#, ,"]','',regex=True)

X1 = "SDP"
X2 = "V_SD_1"
Y = "Isd_DC"

VST_sweep = np.unique(np.array(data_df[X1]))
VSD_sweep = np.unique(np.array(data_df[X2]))
ISD_2D = np.rot90(
    np.array(data_df[Y]).reshape(len(VSD_sweep),len(VST_sweep)),0
)

ISD_1D = ISD_2D.T[:,7]
VSD_1D = VSD_sweep[7]

VST_max, G_max = Palpatine.get_VST_for_Gmax(VST_sweep, ISD_1D, VSD_1D, plot=False)

Palpatine.get_lever_arms(
    VST_sweep, 
    VSD_sweep, 
    ISD_2D, 
    VST_window=(300., 325.), 
    VSD_window=(0., 800.),
    automated=False
)
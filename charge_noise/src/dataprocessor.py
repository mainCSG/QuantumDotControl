import qcodes as qc
import numpy as np

class DataProcessor:
    def __init__(self) -> None:
        pass

    def parse_db_file(self, db_file_path: str, run_id: list, is1Dsweep=False, is2Dsweep=False):
        
        qc.initialise_or_create_database_at(db_file_path)

        if len(run_id) == 1:

            dataset = qc.load_by_run_spec(captured_run_id=run_id[0])
            params = dataset.parameters.split(',')
            
            if is1Dsweep:
                xname, zname = params
                x = np.array(dataset.get_parameter_data(xname)[xname][xname])
                z = np.array(dataset.get_parameter_data(zname)[zname][zname])
                return x, z

            elif is2Dsweep:
                xname, yname, zname = params
                x = np.array(dataset.get_parameter_data(xname)[xname][xname])
                y = np.array(dataset.get_parameter_data(yname)[yname][yname])
                z = np.array(dataset.get_parameter_data(zname)[zname][zname])

                x, y = np.unique(x), y
                return x,y,z
            
        elif len(run_id) == 2:
            # averaging needs to be done

            if is1Dsweep:
                run_id_sweep = [i for i in range(run_id[0], run_id[1]+1)]

                datasets = [qc.load_by_run_spec(captured_run_id=run) for run in run_id_sweep]
                
                params = datasets[0].parameters.split(',')

                xname, zname = params
                x = np.array([datasets[index].get_parameter_data(xname)[xname][xname] for index in range(len(datasets))])[0]
                z = np.array([datasets[index].get_parameter_data(zname)[zname][zname] for index in range(len(datasets))])
                
                z = np.mean(z, axis=0)

                return x, z
        
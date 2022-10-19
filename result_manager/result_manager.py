import os
import dill
import matplotlib
import numpy as np
import pandas as pd
import torch
import yaml

class ResultManager():
    """
    Class to manage any kind of result python object. Will store results in a central location.
    Uses dill as a backend which is more powerful than pickle itself, but will save files in pickle format.
    It allows to load classes that have been modified which is not possible with pickle.
    """

    def __init__(self, root, verbose=True) -> None:
        self.root = root
        self.verbose = verbose

        if not os.path.exists(self.root):
            os.makedirs(self.root)

    def _print(self, string:str):
        if self.verbose:
            print(string)


    def save_model(self, model, filename:str, path:str=None, overwrite:bool=False):
        if path is None:
            path = self.root
        
        path = os.path.join(path, filename)

        if not overwrite and os.path.exists(path):
            self._print(f"Result file {path} already exists and will not be overwritten!")
            return

        torch.save(model.state_dict(), path)

        self._print(f"Model successfully saved to {path}!")

    def save_result(self, result, filename:str, path:str=None, overwrite:bool=False):

        if path is None:
            path = self.root
        
        path = os.path.join(path, filename)

        if not overwrite and os.path.exists(path):
            self._print(f"Result file {path} already exists and will not be overwritten!")
            return

        if type(result) == np.ndarray:
            np.savetxt(path, result)

        elif type(result) == pd.DataFrame:
            result: pd.DataFrame
            pd.to_pickle(obj=result, filepath_or_buffer=path)
        elif filename.endswith('.yml') or filename.endswith('yaml'):
            with open(path, 'w') as stream:
                yaml.dump(data=result, stream=stream)
        elif issubclass(type(result), torch.nn.Module):
            return self.save_model(model=result, filename=filename, path=path, overwrite=overwrite)
        else:
            with open(path, 'wb+') as f:
                dill.dump(result, f)

        self._print(f"Result successfully saved to {path}!")

    def load_result(self, filename:str, path:str=None):

        if path is None:
            path = self.root
        
        path = os.path.join(path, filename)

        if not os.path.exists(path):
            self._print(f"Result file {path} not found.")
            return None

        with open(path, 'rb') as f:
            result = dill.load(f)

        return result

    def save_pdf(self, figs:list, filename:str, path:str=None):
        if path is None:
            path = self.root
        
        path = os.path.join(path, filename)

        pdf = matplotlib.backends.backend_pdf.PdfPages(path)

        for fig in figs:
            pdf.savefig(fig)

        pdf.close()



        
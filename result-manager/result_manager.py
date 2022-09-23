import os
import dill
import matplotlib

class ResultManager():
    """
    Class to manage any kind of result python object. Will store results in a central location.
    Uses dill as a backend which is more powerful than pickle itself, but will save files in pickle format.
    It allows to load classes that have been modified which is not possible with pickle.
    """

    def __init__(self, root = 'results', verbose=True) -> None:
        self.root = root
        self.verbose = verbose

        if not os.path.exists(self.root):
            os.makedirs(self.root)

    def _print(self, string:str):
        if self.verbose:
            print(string)


    def save_result(self, result, filename:str, path:str=None, overwrite:bool=False):

        if path is None:
            path = self.root
        
        path = os.path.join(path, filename)

        if not overwrite and os.path.exists(path):
            self._print(f"Result file {path} already exists and will not be overwritten!")
            return

        with open(path, 'wb+') as f:
            dill.dump(result, f)

        self._print(f"Result successfully save to {path}!")

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



        
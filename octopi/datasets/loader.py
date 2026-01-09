from copick.util.uri import resolve_copick_objects
from monai.transforms import MapTransform
import copick

class LoadCopickd(MapTransform):
    def __init__(self, keys=("image", "label"),
                 root="root", run="run",
                 vol_uri="vol_uri", target_uri="target_uri"):
        super().__init__(keys)
        self.root_key = root
        self.run_key = run
        self.vol_uri_key = vol_uri
        self.target_uri_key = target_uri
        self._root_cache = {}

    def _get_root(self, root_path):
        if root_path not in self._root_cache:
            self._root_cache[root_path] = copick.from_file(root_path)
        return self._root_cache[root_path]

    def __call__(self, data):
        d = dict(data)
        root = self._get_root(d[self.root_key])
        run_name = d[self.run_key]

        vol = resolve_copick_objects(d[self.vol_uri_key], root, "tomogram", run_name=run_name)
        target = resolve_copick_objects(d[self.target_uri_key], root, "segmentation", run_name=run_name)

        d["image"] = vol[0].numpy()
        d["label"] = target[0].numpy()
        return d


class LoadCopickPredictd(MapTransform):
    """
    Loads a tomogram into d["image"] from (root, run, vol_uri).
    Keeps run metadata so you can save outputs back to the right run.
    """
    def __init__(self, keys=("image", ),
                 root="root", run="run",
                 vol_uri="vol_uri"):
        super().__init__(keys)
        self.root_key = root
        self.run_key = run
        self.vol_uri_key = vol_uri
        self._root_cache = {}

    def _get_root(self, root_path):
        if root_path not in self._root_cache:
            self._root_cache[root_path] = copick.from_file(root_path)
        return self._root_cache[root_path]

    def __call__(self, data):
        d = dict(data)
        root = self._get_root(d[self.root_key])
        run_name = d[self.run_key]
        vol = resolve_copick_objects(
            d[self.vol_uri_key], 
            root, 
            "tomogram", 
            run_name=run_name
        )[0].numpy()
        
        d["image"] = vol[0].numpy()
        return d
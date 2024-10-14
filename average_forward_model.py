import mne
import pathlib
import numpy as np
from collections import OrderedDict


class boundary_element_model_config:
    """
    Configuration class for the boundary element model (BEM).

    Attributes:
        icosahedral_subdivision_level (int): The resolution of the cortical surface mesh.
            Specifies the surface downsampling level. For example:
            - 5 corresponds to 20484 vertices,
            - 4 corresponds to 5120 vertices,
            - 3 corresponds to 1280 vertices.
            If set to None, no subsampling is applied.
        conductivities_of_the_layers (tuple): A tuple representing the conductivities for each shell.
            It can be:
            - A single float for a one-layer model (e.g., (0.3,)).
            - Three elements for a three-layer model (e.g., (0.3, 0.006, 0.3)).
            Default is (0.3, 0.006, 0.3).
    """

    def __init__(
        self,
        icosahedral_subdivision_level: int,
        conductivities_of_the_layers: tuple,
    ):
        self.icosahedral_subdivision_level = icosahedral_subdivision_level
        self.conductivities_of_the_layers = conductivities_of_the_layers


default_bem_config = boundary_element_model_config(
    icosahedral_subdivision_level=4,
    conductivities_of_the_layers=(
        0.3,
        0.006,
        0.3,
    ),
)


class source_space_config:
    """
    Configuration class for the source space.

    Attributes:
        spacing_between_dipoles (float): Defines the positions of sources using grid spacing in mm.
        minimal_distance_of_source_from_inner_skull (float): Excludes points closer than this distance (in mm) to the bounding surface.
    """

    def __init__(
        self,
        spacing_between_dipoles: float,
        minimal_distance_of_source_from_inner_skull: float,
    ):
        self.spacing_between_dipoles = spacing_between_dipoles
        self.minimal_distance_of_source_from_inner_skull = minimal_distance_of_source_from_inner_skull


default_ss_config = source_space_config(
    spacing_between_dipoles=5, minimal_distance_of_source_from_inner_skull=5.0
)


class average_forward_model_config:
    """
    Configuration class for the average forward model.

    Attributes:
        montage_name (str): The name of the montage, e.g., biosemi64.
        bem (boundary_element_model_config): Configuration of the boundary element model.
        source (source_space_config): Configuration of the source space.
        njobs: Execution policy.
    """

    def __init__(
        self,
        montage_name: str,
        bem: boundary_element_model_config,
        source: source_space_config,
        njobs,
    ):
        self.montage_name = montage_name
        self.boundary_element_model = bem
        self.source = source
        self.njobs = njobs


default_biosemi64_fm_config = average_forward_model_config(
    montage_name="biosemi64",
    bem=default_bem_config,
    source=default_ss_config,
    njobs=10,
)


class average_forward_model(mne.Forward):
    """
    Class for creating and managing an average forward model.

    Attributes:
        config (average_forward_model_config): Configuration for the average forward model.
        path (pathlib.Path): Path in which this average model lives.
        montage (mne.DigMontage): The montage configuration.
    """

    def __init__(self, config: average_forward_model_config):
        self.config = config
        self.path = pathlib.Path(mne.get_config("SUBJECTS_DIR"))
        mne.datasets.fetch_fsaverage(subjects_dir=None)
        print(f"path = {self.path}")
        self.__create_source_space()
        self.__create_boundary_element_solution()
        self.montage: mne.DigMontage = mne.channels.make_standard_montage(
            self.config.montage_name
        )
        self.__create_info_for_forward_solution()
        super().__init__(self.__create_forward_solution())

    def save(self, path: pathlib.Path):
        """
        Saves the forward model to the specified path.

        Parameters:
            path (pathlib.Path): The path to save the forward model.
        """
        path.mkdir(parents=True, exist_ok=True)
        super().save(path, overwrite=True, verbose=True)

    def __create_source_space(self):
        """
        Creates the source space for the forward model.

        Utilizes the `mne.setup_volume_source_space` function to generate
        a set of points that represent the cortical surface of the brain.
        This is essential for modeling how brain activity, captured through
        MEG or EEG, corresponds to specific locations on the cortex.

        Sampling Points:
        It allows users to specify various algorithms for sampling points
        on the cortical surface, such as using an icosahedral representation
        (e.g., --ico -6), which determines the density of points. This
        influences the spatial resolution of the source localization.

        Integration with Freesurfer:
        The source space is often created based on anatomical information
        derived from Freesurfer, which segments the brain's anatomy and
        provides surfaces that MNE can utilize. The output file typically
        contains coordinates for these points in a format compatible with
        further analysis in MNE.

        Facilitate Source Localization:
        By creating a detailed source space, researchers can more accurately
        localize brain activity sources during analysis, allowing for better
        interpretation of neuroimaging results.

        Output File Generation:
        The function generates output files (e.g., .fif format) that can be
        used in subsequent analyses, such as inverse modeling or connectivity
        analysis.
        """
        surface_dir: pathlib.Path = (
            self.path / "fsaverage" / "bem" / "inner_skull.surf"
        )
        self.source_space: mne.SourceSpaces = mne.setup_volume_source_space(
            "fsaverage",
            subjects_dir=self.path,
            surface=surface_dir,
            add_interpolator=False,
            pos=self.config.source.spacing_between_dipoles,
            n_jobs=self.config.njobs,
        )

    def __create_boundary_element_solution(self):
        """
        Creates the boundary element solution for the forward model.

        Uses the `mne.make_bem_model` and `mne.make_bem_solution` functions
        to generate the BEM model and solution.
        """
        bem = self.config.boundary_element_model
        model = mne.make_bem_model(
            subject="fsaverage",
            ico=bem.icosahedral_subdivision_level,
            conductivity=bem.conductivities_of_the_layers,
            subjects_dir=self.path,
        )
        self.boundary_element_solution: mne.bem.ConductorModel = (
            mne.make_bem_solution(model)
        )

    def __update_channel_positions(
        self, info: mne.Info, channel_position: OrderedDict
    ):
        """
        Updates the positions of the channels in the info object.

        Parameters:
            info (mne.Info): The info object containing channel information.
            channel_position (OrderedDict): The new positions of the channels.
        """
        for channel in info["chs"]:
            name = channel["ch_name"]
            position = channel_position[name]
            channel["loc"][0:3] = position

    def __create_info_for_forward_solution(self) -> mne.Info:
        """
        Creates the info object for the forward solution.

        Returns:
            mne.Info: The info object.
        """
        sampling_rate = 1  # should be ignored when forward model is created
        self.metadata_information: mne.Info = mne.create_info(
            self.montage.ch_names, sampling_rate, ch_types="eeg"
        )
        self.__update_channel_positions(
            self.metadata_information, self.montage._get_ch_pos()
        )

    def __create_forward_solution(self) -> mne.Forward:
        """
        Creates the forward solution.

        Returns:
            mne.Forward: The forward solution object.
        """
        transformation_file: pathlib.Path = (
            self.path / "fsaverage" / "bem" / "fsaverage-trans.fif"
        )
        return mne.make_forward_solution(
            self.metadata_information,
            trans=transformation_file,
            src=self.source_space,
            bem=self.boundary_element_solution,
            meg=False,
            eeg=True,
            mindist=self.config.source.minimal_distance_of_source_from_inner_skull,
            n_jobs=self.config.njobs,
            verbose=True,
        )


afm = average_forward_model(default_biosemi64_fm_config)

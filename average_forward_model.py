import mne
import pathlib
import numpy as np
from collections import OrderedDict


class boundary_element_model_config:
    def __init__(
        self,
        icosahedral_subdivision_level: int,
        conductivities_of_the_layers: tuple,
    ):
        # The resolution of the cortical surface mesh Specifies the surface downsampling level. For example:
        # 5 corresponds to 20484 vertices,
        # 4 corresponds to 5120 vertices,
        # 3 corresponds to 1280 vertices.
        # If set to None, no subsampling is applied.
        self.icosahedral_subdivision_level = icosahedral_subdivision_level
        # A tuple representing the conductivities for each shell. It can be:
        # A single float for a one-layer model (e.g., (0.3,)).
        # Three elements for a three-layer model (e.g., (0.3, 0.006, 0.3)).
        # Default is (0.3, 0.006, 0.3).
        self.conductivities_of_the_layers = conductivities_of_the_layers


# why so???
default_bem_config = boundary_element_model_config(
    icosahedral_subdivision_level=4,
    conductivities_of_the_layers=(
        0.3,
        0.006,
        0.3,
    ),
)


class source_space_config:
    def __init__(
        self,
        spacing_between_dipoles: float,
        minimal_distance_of_source_from_inner_skull: float,
    ):
        # Defines the positions of sources using  grid spacing in mm
        self.spacing_between_dipoles = spacing_between_dipoles
        # Excludes points closer than this distance (in mm) to the bounding surface.
        self.minimal_distance_of_source_from_inner_skull = (
            minimal_distance_of_source_from_inner_skull
        )


# why so???
default_ss_config = source_space_config(
    spacing_between_dipoles=5, minimal_distance_of_source_from_inner_skull=5.0
)


class average_forward_model_config:
    def __init__(
        self,
        # path: pathlib.Path,
        montage_name: str,
        bem: boundary_element_model_config,
        source: source_space_config,
        njobs,
    ):
        # The path in which this average model lives
        # self.path = path
        # The name of montage e.g. biosemi64
        self.montage_name = montage_name
        # configuration of boundary element model
        self.boundary_element_model = bem
        # configuration of source space
        self.source = source
        # execution policy
        self.njobs = njobs


default_biosemi64_fm_config = average_forward_model_config(
    # path="./models/forward/average",
    montage_name="biosemi64",
    bem=default_bem_config,
    source=default_ss_config,
    njobs=10,
)


class average_forward_model(mne.Forward):
    # General configuration of forward mode includeing monatage name,
    # boundary element model configuration, source configuration and
    # execution policy that is njobs
    # config:average_forward_model_config
    # source_space: mne.SourceSpaces
    # boundary_element_solution: mne.bem.ConductorModel
    # metadata_information: mne.Info
    def __init__(self, config: average_forward_model_config):
        self.config = config
        # designed to download and prepare the "fsaverage" brain template,
        # which is a widely used standard anatomical model in neuroimaging.
        # This template is particularly useful for visualizing and analyzing
        # data across different subjects by providing a common reference.
        # self.path = mne.datasets.fetch_fsaverage(subjects_dir=self.path)
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
        path.mkdir(parents=True, exist_ok=True)
        super().save(path, overwrite=True, verbose=True)

    # The function generates a set of points that represent the cortical surface of the brain. This is essential for modeling how brain activity, captured through MEG or EEG, corresponds to specific locations on the cortex.
    # Sampling Points:
    # It allows users to specify various algorithms for sampling points on the cortical surface, such as using an icosahedral representation (e.g., --ico -6), which determines the density of points. This influences the spatial resolution of the source localization.
    # Integration with Freesurfer:
    # The source space is often created based on anatomical information derived from Freesurfer, which segments the brain's anatomy and provides surfaces that MNE can utilize. The output file typically contains coordinates for these points in a format compatible with further analysis in MNE.
    # Facilitate Source Localization:
    # By creating a detailed source space, researchers can more accurately localize brain activity sources during analysis, allowing for better interpretation of neuroimaging results.
    # Output File Generation:
    # The function generates output files (e.g., .fif format) that can be used in subsequent analyses, such as inverse modeling or connectivity analysis.
    def __create_source_space(self):
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
        bem = self.config.boundary_element_model
        model = mne.make_bem_model(
            subject="fsaverage",
            ico=bem.icosahedral_subdivision_level,
            conductivity=bem.conductivities_of_the_layers,
            subjects_dir=self.path,
        )
        #  why this type is not recognized?
        # https://mne.tools/stable/generated/mne.bem.ConductorModel.html#mne.bem.ConductorModel
        self.boundary_element_solution: mne.bem.ConductorModel = (
            mne.make_bem_solution(model)
        )

    def __update_channel_positions(
        self, info: mne.Info, channel_position: OrderedDict
    ):
        for channel in info["chs"]:
            name = channel["ch_name"]
            position = channel_position[name]
            channel["loc"][0:3] = position

    def __create_info_for_forward_solution(self) -> mne.Info:
        sampling_rate = 1  # should be ignored when forward model is created?? Is there a way to not hack mne?
        self.metadata_information: mne.Info = mne.create_info(
            self.montage.ch_names, sampling_rate, ch_types="eeg"
        )
        self.__update_channel_positions(
            self.metadata_information, self.montage._get_ch_pos()
        )

    def __create_forward_solution(self) -> mne.Forward:
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
#
#


# mne.datasets.fetch_fsaverage(subjects_dir=None)


# def save_forward_model(
#     fwd: mne.Forward,
#     destination_directory: pathlib.Path,
#     standard_montage_name: str,
#     grid_size_mm: float,
# ):
#     save_directory = destination_directory / "forward_solutions"
#     save_directory.mkdir(parents=True, exist_ok=True)
#     save_file = (
#         save_directory
#         / f"average_{standard_montage_name}_grid_{grid_size_mm}_mm-fwd.fif"
#     )
#     fwd.save(save_file, overwrite=True, verbose=True)


# def fetch_average_eeg_forward_model(
#     destination_directory: pathlib.Path,
#     standard_montage_name: str,
#     grid_size_mm: float,
# ) -> mne.Forward:
#     destination_directory.mkdir(parents=True, exist_ok=True)
#     source_space = create_source_space(destination_directory, grid_size_mm)
#     bem_solution = create_boundary_element_method_solution(
#         destination_directory
#     )
#     print(f"\n\n\n BEM SOLUTION = {bem_solution}\n\n\n")
#     info = create_info_for_forward_solution(
#         destination_directory, standard_montage_name
#     )
#     fwd = create_forward_solution(
#         destination_directory, info, source_space, bem_solution, grid_size_mm
#     )
#     save_forward_model(
#         fwd, destination_directory, standard_montage_name, grid_size_mm
#     )

#     return fwd


# fetch_average_eeg_forward_model(
#     pathlib.Path("./data/average/"),
#     "biosemi64",
#     5,
# )

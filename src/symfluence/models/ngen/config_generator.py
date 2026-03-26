# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
NGEN Model Configuration Generator

Handles generation of model-specific configuration files for NextGen Framework:
- CFE (Conceptual Functional Equivalent) configs
- PET (Potential Evapotranspiration) configs
- NOAH-OWP (Noah-Owens-Pries) configs
- Snow-17 configuration files (Fortran NAMELIST format)
- SACSMA (Sacramento Soil Moisture Accounting) configs

SPECIAL NOTE: Snow-17 Configuration Format (v2.0 - BREAKING CHANGE)
=====================================================================

Snow-17's Fortran BMI wrapper requires strict Fortran NAMELIST format for ALL inputs.
This was NOT enforced in prior versions, causing "End of file" runtime errors.

**OLD FORMAT (BROKEN):**
  cat-{id}_snow17_config.txt (single file)
  - Contained only snowpack parameters (SCF, PXTEMP, MFMAX, ...)
  - Plain key=value format
  - MISSING control parameters (timestep, file paths, dates)
  - Result: Snow-17 Fortran module fails with "End of file" error

**NEW FORMAT (REQUIRED):**
  TWO separate files per catchment:
  1. cat-{id}_snow17_config.txt (CONTROL NAMELIST)
     - Fortran NAMELIST: &SNOW17_CONTROL ... /
     - Contains control parameters (model_timestep, dates, file paths)
     - Loaded by Snow-17's namelistModule.f90::readNamelist()
     - Referenced in NGEN realization JSON
  
  2. cat-{id}_snow17_parameters.txt (SNOWPACK PARAMETERS)
      - Plain key value format for snowpack calibration parameters
     - Referenced by snow17_param_file in control namelist
     - Supports multi-HRU format (extensible)

**MIGRATION GUIDE:**
- Old configs auto-detected and logged as deprecated (non-fatal warning)
- Recommended: Re-run geoprocessing step via `sf workflow run` to auto-regenerate
- No manual fixes needed; new configs generated automatically

**BACKWARDS COMPATIBILITY:**
- detect_and_migrate_snow17_format() method checks for deprecated format
- Old format silently detected and warned; still proceeds (may fail at NGEN runtime)
- Next preprocessing run will generate correct format

References:
- Snow-17 Source: github.com/NOAA-OWP/snow17 (src/share/namelistModule.f90)
- NWS SNOW-17 Model Documentation: https://www.nohrsc.noaa.gov/nwsrfs/

Extracted from NgenPreProcessor to improve modularity and testability.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import geopandas as gpd
import pandas as pd

from symfluence.core.mixins import ConfigMixin


class NgenConfigGenerator(ConfigMixin):
    """
    Generator for NGEN model configuration files.

    Handles creation of:
    - CFE BMI configuration files (.txt)
    - PET configuration files (.txt)
    - NOAH-OWP input files (.input)
    - Realization configuration (JSON)

    Attributes:
        config: Configuration dictionary
        logger: Logger instance
        setup_dir: Path to NGEN settings directory
        catchment_crs: CRS of the catchment geometries
    """

    def __init__(
        self,
        config: Dict[str, Any],
        logger: Any,
        setup_dir: Path,
        catchment_crs: Any = None
    ):
        """
        Initialize the config generator.

        Args:
            config: Configuration dictionary
            logger: Logger instance
            setup_dir: Path to NGEN settings directory
            catchment_crs: Coordinate reference system for catchments
        """
        from symfluence.core.config.coercion import coerce_config
        self._config = coerce_config(config, warn=False)
        self.logger = logger
        self.setup_dir = Path(setup_dir)
        self.catchment_crs = catchment_crs

        # Module availability (set by preprocessor)
        self._include_cfe = True
        self._include_pet = True
        self._include_noah = True
        self._include_sloth = False
        self._include_topmodel = False
        self._include_sacsma = False
        self._include_snow17 = False
        self._noah_et_fallback = 'EVAPOTRANS'  # Default when PET disabled but NOAH enabled

    def set_module_availability(
        self,
        cfe: bool = True,
        pet: bool = True,
        noah: bool = True,
        sloth: bool = False,
        noah_et_fallback: str = 'EVAPOTRANS',
        topmodel: bool = False,
        sacsma: bool = False,
        snow17: bool = False,
    ):
        """Set which NGEN modules are available.

        Args:
            cfe: Enable CFE runoff module
            pet: Enable PET evapotranspiration module
            noah: Enable NOAH-OWP land surface module
            sloth: Enable SLOTH ice fraction module
            noah_et_fallback: When PET disabled but NOAH enabled, which NOAH output
                to use as the runoff module's ET input. Options:
                - 'EVAPOTRANS': Total evapotranspiration rate [m/s] (default)
                - 'ETRAN': Transpiration [mm] - WRONG UNITS
                - 'QSEVA': Evaporation rate [mm/s] - rate but wrong dimension
            topmodel: Enable TOPMODEL runoff module (alternative to CFE)
            sacsma: Enable SAC-SMA runoff module (alternative to CFE)
            snow17: Enable Snow-17 snow module (alternative to NOAH snow physics)
        """
        self._include_cfe = cfe
        self._include_pet = pet
        self._include_noah = noah
        self._include_sloth = sloth
        self._include_topmodel = topmodel
        self._include_sacsma = sacsma
        self._include_snow17 = snow17
        self._noah_et_fallback = noah_et_fallback

    def generate_all_configs(
        self,
        catchment_gdf: gpd.GeoDataFrame,
        hru_id_col: str
    ) -> None:
        """
        Generate all model configuration files for each catchment.

        Args:
            catchment_gdf: GeoDataFrame with catchment geometries
            hru_id_col: Column name for catchment IDs
        """
        self.logger.info("Generating model configuration files")
        self.catchment_crs = catchment_gdf.crs

        for idx, catchment in catchment_gdf.iterrows():
            cat_id = str(catchment[hru_id_col])

            if self._include_cfe:
                self.generate_cfe_config(cat_id, catchment)
            if self._include_topmodel:
                self.generate_topmodel_config(cat_id, catchment)
            if self._include_sacsma:
                self.generate_sacsma_config(cat_id, catchment)
            if self._include_pet:
                self.generate_pet_config(cat_id, catchment)
            if self._include_noah:
                self.generate_noah_config(cat_id, catchment)
            if self._include_snow17:
                self.generate_snow17_config(cat_id, catchment)

        self.logger.info(f"Generated configs for {len(catchment_gdf)} catchments")

    def generate_cfe_config(
        self,
        catchment_id: str,
        catchment_row: Optional[gpd.GeoSeries] = None,
        **overrides
    ) -> Path:
        """
        Generate CFE model configuration file.

        Args:
            catchment_id: Catchment identifier
            catchment_row: Optional GeoSeries with catchment data
            **overrides: Parameter overrides (e.g., soil_b=6.0)

        Returns:
            Path to generated config file
        """
        # Default parameters (can be overridden)
        params = {
            'depth': 2.0,
            'soil_b': 5.0,
            'satdk': 5.0e-06,
            'satpsi': 0.141,
            'slop': 0.03,
            'smcmax': 0.439,
            'wltsmc': 0.047,
            'expon': 1.0,
            'expon_secondary': 1.0,
            'refkdt': 1.0,
            'max_gw_storage': 0.2,
            'cgw': 1.8e-05,
            'gw_expon': 7.0,
            'gw_storage': 0.35,
            'alpha_fc': 0.33,
            'soil_storage': 0.35,
            'k_nash': 0.03,
            'k_lf': 0.01,
        }

        # Apply overrides
        params.update(overrides)

        # Calculate catchment area in km² (required for CFE to convert depth to m³/s)
        catchment_area_km2 = 1.0  # Default fallback
        if catchment_row is not None:
            # Try to get area from pre-computed column first
            # Columns explicitly in km²: areasqkm, area_km2
            # Columns that may be in m²: HRU_area, GRU_area, AREA
            km2_cols = ['areasqkm', 'area_km2']  # Known to be in km²
            m2_cols = ['HRU_area', 'GRU_area', 'AREA']  # Often in m², need conversion check

            for area_col in km2_cols + m2_cols:
                if area_col in catchment_row.index:
                    area_val = catchment_row[area_col]
                    if area_val is not None and pd.notna(area_val) and float(area_val) > 0:
                        area_val = float(area_val)
                        # Check if value is likely in m² (no catchment is > 1 million km²)
                        # If value > 1e6, assume it's in m² and convert to km²
                        if area_col in m2_cols and area_val > 1e6:
                            catchment_area_km2 = area_val / 1e6
                            self.logger.debug(f"Converted {area_col} from m² to km²: {catchment_area_km2:.2f} km²")
                        else:
                            catchment_area_km2 = area_val
                            self.logger.debug(f"Using area from {area_col}: {catchment_area_km2:.2f} km²")
                        break
            else:
                # Calculate from geometry if no area column found
                if hasattr(catchment_row, 'geometry') and catchment_row.geometry is not None:
                    geom = catchment_row.geometry
                    # Project to equal-area CRS for accurate area calculation
                    if self.catchment_crs is not None and self.catchment_crs.is_geographic:
                        # Use UTM zone based on centroid for local accuracy
                        import pyproj
                        centroid = geom.centroid
                        utm_zone = int((centroid.x + 180) / 6) + 1
                        hemisphere = 'north' if centroid.y >= 0 else 'south'
                        utm_crs = pyproj.CRS(f"+proj=utm +zone={utm_zone} +{hemisphere} +datum=WGS84")
                        transformer = pyproj.Transformer.from_crs(
                            self.catchment_crs, utm_crs, always_xy=True
                        )
                        from shapely.ops import transform
                        geom_projected = transform(transformer.transform, geom)
                        catchment_area_km2 = geom_projected.area / 1e6
                    else:
                        # Assume already in projected CRS with meters
                        catchment_area_km2 = geom.area / 1e6
                    self.logger.debug(f"Calculated area from geometry: {catchment_area_km2:.2f} km²")

        if catchment_area_km2 <= 0:
            self.logger.warning(
                f"Invalid catchment area ({catchment_area_km2}) for cat-{catchment_id}, "
                f"using default 1.0 km². CFE Q_OUT will be in depth units (m), not m³/s!"
            )
            catchment_area_km2 = 1.0

        # Calculate num_timesteps using configured forcing timestep (not hardcoded hourly)
        start_time = self._get_config_value(lambda: self.config.domain.time_start, default='2000-01-01 00:00:00', dict_key='EXPERIMENT_TIME_START')
        end_time = self._get_config_value(lambda: self.config.domain.time_end, default='2000-12-31 23:00:00', dict_key='EXPERIMENT_TIME_END')
        if start_time == 'default': start_time = '2000-01-01 00:00:00'
        if end_time == 'default': end_time = '2000-12-31 23:00:00'

        # Get forcing timestep from config (default 3600s = 1 hour)
        forcing_timestep = self._get_config_value(lambda: self.config.forcing.time_step_size, default=3600, dict_key='FORCING_TIME_STEP_SIZE')
        try:
            forcing_timestep = int(forcing_timestep)
        except (ValueError, TypeError):
            forcing_timestep = 3600

        try:
            duration = pd.to_datetime(end_time) - pd.to_datetime(start_time)
            # Use configured timestep, add 1 for inclusive end bound
            num_steps = int(duration.total_seconds() / forcing_timestep) + 1
        except (ValueError, TypeError):
            num_steps = 1

        config_text = f"""forcing_file=BMI
catchment_area_km2={catchment_area_km2}[km2]
soil_params.depth={params['depth']}[m]
soil_params.b={params['soil_b']}[]
soil_params.satdk={params['satdk']:.2e}[m s-1]
soil_params.satpsi={params['satpsi']}[m]
soil_params.slop={params['slop']}[m/m]
soil_params.smcmax={params['smcmax']}[m/m]
soil_params.wltsmc={params['wltsmc']}[m/m]
soil_params.expon={params['expon']}[]
soil_params.expon_secondary={params['expon_secondary']}[]
max_gw_storage={params['max_gw_storage']}[m]
Cgw={params['cgw']:.2e}[m h-1]
expon={params['gw_expon']}[]
gw_storage={params['gw_storage']}[m/m]
alpha_fc={params['alpha_fc']}[]
soil_storage={params['soil_storage']}[m/m]
K_nash={params['k_nash']}[]
K_lf={params['k_lf']}[]
nash_storage=0.0,0.0
giuh_ordinates={self._generate_giuh_ordinates(catchment_area_km2)}
num_timesteps={num_steps}
verbosity=1
surface_runoff_scheme=GIUH
surface_water_partitioning_scheme=Schaake
"""

        config_file = self.setup_dir / "CFE" / f"cat-{catchment_id}_bmi_config_cfe_pass.txt"
        config_file.parent.mkdir(parents=True, exist_ok=True)
        with open(config_file, 'w', encoding='utf-8') as f:
            f.write(config_text)

        return config_file

    def generate_pet_config(
        self,
        catchment_id: str,
        catchment_row: gpd.GeoSeries,
        **overrides
    ) -> Path:
        """
        Generate PET model configuration file.

        Args:
            catchment_id: Catchment identifier
            catchment_row: GeoSeries with catchment geometry
            **overrides: Parameter overrides

        Returns:
            Path to generated config file
        """
        # Get catchment centroid
        centroid = self._get_wgs84_centroid(catchment_row)

        # Extract elevation from catchment attributes if available
        catchment_elevation = None
        for elev_attr in ['elevation_m', 'elevation', 'elev_mean', 'mean_elev', 'elev_m']:
            if elev_attr in catchment_row.index and pd.notna(catchment_row[elev_attr]):
                catchment_elevation = float(catchment_row[elev_attr])
                break

        # Get config-level PET settings
        ngen_config = self._get_config_value(lambda: self.config.model.ngen, default=None)
        pet_config = (getattr(ngen_config, 'pet', None) or {}) if ngen_config and not isinstance(ngen_config, dict) else (ngen_config or {}).get('PET', {})

        # Use config elevation if specified, otherwise catchment attribute, otherwise default
        elevation = pet_config.get('elevation_m', catchment_elevation if catchment_elevation else 100.0)

        # Set vegetation parameters based on elevation (forest vs grass)
        # Mountain watersheds (>1000m) typically have forest/mixed vegetation
        if elevation > 1000:
            default_veg_height = 15.0  # Forest
            default_zero_plane = 10.0
            default_momentum_roughness = 1.5
        else:
            default_veg_height = 0.12  # Grass
            default_zero_plane = 0.0003
            default_momentum_roughness = 0.0

        # Default parameters with intelligent defaults
        # Use forcing timestep for PET instead of hardcoded hourly
        forcing_timestep = self._get_config_value(lambda: self.config.forcing.time_step_size, default=3600, dict_key='FORCING_TIME_STEP_SIZE')
        try:
            forcing_timestep = int(forcing_timestep)
        except (ValueError, TypeError):
            forcing_timestep = 3600

        params = {
            'wind_height': pet_config.get('wind_height_m', 10.0),
            'humidity_height': pet_config.get('humidity_height_m', 2.0),
            'veg_height': pet_config.get('vegetation_height_m', default_veg_height),
            'zero_plane_displacement': pet_config.get('zero_plane_displacement_m', default_zero_plane),
            'momentum_roughness': pet_config.get('momentum_roughness_length_m', default_momentum_roughness),
            'heat_roughness': pet_config.get('heat_roughness_length_m', 0.0),
            'emissivity': pet_config.get('emissivity', 1.0),
            'albedo': pet_config.get('albedo', 0.23),
            'elevation': elevation,
            'timestep': forcing_timestep,
        }
        params.update(overrides)

        # Log PET configuration parameters
        self.logger.info(f"Generating PET config for {catchment_id}:")
        self.logger.info(f"  Elevation: {params['elevation']:.1f} m" +
                        (" (from catchment attribute)" if catchment_elevation else " (default)"))
        self.logger.info(f"  Vegetation height: {params['veg_height']:.2f} m")
        self.logger.info(f"  Momentum roughness: {params['momentum_roughness']:.2f} m")

        # Calculate num_timesteps using configured forcing timestep (already in params['timestep'])
        start_time = self._get_config_value(lambda: self.config.domain.time_start, default='2000-01-01 00:00:00', dict_key='EXPERIMENT_TIME_START')
        end_time = self._get_config_value(lambda: self.config.domain.time_end, default='2000-12-31 23:00:00', dict_key='EXPERIMENT_TIME_END')
        if start_time == 'default': start_time = '2000-01-01 00:00:00'
        if end_time == 'default': end_time = '2000-12-31 23:00:00'

        try:
            duration = pd.to_datetime(end_time) - pd.to_datetime(start_time)
            # Use configured timestep (forcing_timestep), add 1 for inclusive end bound
            num_steps = int(duration.total_seconds() / forcing_timestep) + 1
        except (ValueError, TypeError):
            num_steps = 1

        config_text = f"""verbose=0
pet_method=5
forcing_file=BMI
run_unit_tests=0
yes_aorc=1
yes_wrf=0
wind_speed_measurement_height_m={params['wind_height']}
humidity_measurement_height_m={params['humidity_height']}
vegetation_height_m={params['veg_height']}
zero_plane_displacement_height_m={params['zero_plane_displacement']}
momentum_transfer_roughness_length={params['momentum_roughness']}
heat_transfer_roughness_length_m={params['heat_roughness']}
surface_longwave_emissivity={params['emissivity']}
surface_shortwave_albedo={params['albedo']}
cloud_base_height_known=FALSE
latitude_degrees={centroid.y}
longitude_degrees={centroid.x}
site_elevation_m={params['elevation']}
time_step_size_s={params['timestep']}
num_timesteps={num_steps}
shortwave_radiation_provided=1
"""

        config_file = self.setup_dir / "PET" / f"cat-{catchment_id}_pet_config.txt"
        config_file.parent.mkdir(parents=True, exist_ok=True)
        with open(config_file, 'w', encoding='utf-8') as f:
            f.write(config_text)

        return config_file

    def generate_noah_config(
        self,
        catchment_id: str,
        catchment_row: gpd.GeoSeries,
        **overrides
    ) -> Path:
        """
        Generate NOAH-OWP model configuration file (.input file).

        Creates a Fortran namelist file with all required sections.

        Args:
            catchment_id: Catchment identifier
            catchment_row: GeoSeries with catchment geometry
            **overrides: Parameter overrides

        Returns:
            Path to generated config file
        """
        # Get catchment centroid
        centroid = self._get_wgs84_centroid(catchment_row)

        # Get simulation timing from config
        start_time = self._get_config_value(lambda: self.config.domain.time_start, default='2000-01-01 00:00:00', dict_key='EXPERIMENT_TIME_START')
        end_time = self._get_config_value(lambda: self.config.domain.time_end, default='2000-12-31 23:00:00', dict_key='EXPERIMENT_TIME_END')

        # Handle 'default' strings
        if start_time == 'default':
            start_time = '2000-01-01 00:00:00'
        if end_time == 'default':
            end_time = '2000-12-31 23:00:00'

        start_dt = pd.to_datetime(start_time)
        end_dt = pd.to_datetime(end_time)
        start_str = start_dt.strftime('%Y%m%d%H%M')
        end_str = end_dt.strftime('%Y%m%d%H%M')

        # Absolute path to parameters directory
        param_dir = str((self.setup_dir / "NOAH" / "parameters").resolve()) + "/"

        # Extract terrain slope from catchment attributes if available
        terrain_slope = 0.01  # Default: gentle slope (more physical than 0.0)
        if catchment_row is not None:
            for slope_attr in ['terrain_slope', 'slope_mean', 'mean_slope', 'SLOPE', 'slope']:
                if slope_attr in catchment_row.index and pd.notna(catchment_row[slope_attr]):
                    terrain_slope = float(catchment_row[slope_attr])
                    self.logger.debug(f"Using terrain slope from {slope_attr}: {terrain_slope}")
                    break

        # Default NOAH options
        options = {
            'dt': 3600.0,
            'terrain_slope': terrain_slope,
            'azimuth': 0.0,
            'zref': 10.0,
            'rain_snow_thresh': 1.0,
            'soil_type': 3,
            'veg_type': 10,
            # Soil moisture parameters for physically-consistent initialization
            'smcmax': 0.5,    # Saturation moisture content (porosity)
            'smcwlt': 0.1,    # Wilting point moisture content
        }
        options.update(overrides)

        # Calculate initial soil moisture as a reasonable fraction of saturation
        # This ensures initial conditions are physically consistent with calibration bounds
        # Default: 60% of saturation (between field capacity and saturation)
        initial_moisture_frac = self._get_config_value(
            lambda: self.config.model.ngen.initial_moisture_fraction,
            default=0.6,
            dict_key='NOAH_INITIAL_MOISTURE_FRACTION'
        )
        initial_sh2o = options['smcmax'] * initial_moisture_frac
        # Ensure we're at least at wilting point
        initial_sh2o = max(initial_sh2o, options['smcwlt'])

        config_text = f"""&timing
  dt                 = {options['dt']}
  startdate          = "{start_str}"
  enddate            = "{end_str}"
  forcing_filename   = "BMI"
  output_filename    = "out_cat-{catchment_id}.csv"
/

&parameters
  parameter_dir      = "{param_dir}"
  general_table      = "GENPARM.TBL"
  soil_table         = "SOILPARM.TBL"
  noahowp_table      = "MPTABLE.TBL"
  soil_class_name    = "STAS"
  veg_class_name     = "MODIFIED_IGBP_MODIS_NOAH"
/

&location
  lat                = {centroid.y}
  lon                = {centroid.x}
  terrain_slope      = {options['terrain_slope']}
  azimuth            = {options['azimuth']}
/

&forcing
  ZREF               = {options['zref']}
  rain_snow_thresh   = {options['rain_snow_thresh']}
/

&model_options
  precip_phase_option               = 1
  snow_albedo_option                = 1
  dynamic_veg_option                = 4
  runoff_option                     = 3
  drainage_option                   = 8
  frozen_soil_option                = 1
  dynamic_vic_option                = 1
  radiative_transfer_option         = 3
  sfc_drag_coeff_option             = 1
  canopy_stom_resist_option         = 1
  crop_model_option                 = 0
  snowsoil_temp_time_option         = 3
  soil_temp_boundary_option         = 2
  supercooled_water_option          = 1
  stomatal_resistance_option        = 1
  evap_srfc_resistance_option       = 4
  subsurface_option                 = 2
/

&structure
 isltyp           = {options['soil_type']}
 nsoil            = 4
 nsnow            = 3
 nveg             = 20
 vegtyp           = {options['veg_type']}
 croptype         = 0
 sfctyp           = 1
 soilcolor        = 4
/

&initial_values
 dzsnso    =  0.0,  0.0,  0.0,  0.1,  0.3,  0.6,  1.0
 sice      =  0.0,  0.0,  0.0,  0.0
 sh2o      =  {initial_sh2o:.3f},  {initial_sh2o:.3f},  {initial_sh2o:.3f},  {initial_sh2o:.3f}
 zwt       =  -2.0
/
"""

        config_file = self.setup_dir / "NOAH" / f"cat-{catchment_id}.input"
        config_file.parent.mkdir(parents=True, exist_ok=True)
        with open(config_file, 'w', encoding='utf-8') as f:
            f.write(config_text)

        return config_file

    def generate_topmodel_config(
        self,
        catchment_id: str,
        catchment_row: Optional[gpd.GeoSeries] = None,
        **overrides
    ) -> Path:
        """Generate TOPMODEL BMI configuration file.

        Args:
            catchment_id: Catchment identifier
            catchment_row: Optional GeoSeries with catchment data
            **overrides: Parameter overrides

        Returns:
            Path to generated config file
        """
        params = {
            'szm': 0.05,         # Transmissivity decay parameter [m]
            't0': 2.0,           # Effective log transmissivity [ln(m²/h)]
            'td': 5.0,           # Unsaturated zone time delay [h/m]
            'srmax': 0.04,       # Max root zone storage [m]
            'sr0': 0.01,         # Initial root zone deficit [m]
            'chv': 1000.0,       # Channel flow velocity [m/h]
            'Q0': 0.0001,        # Initial subsurface flow [m/h]
        }
        params.update(overrides)

        # Generate synthetic topographic index distribution
        # (mean TI ~7, std ~3, discretized into 30 classes)
        import math
        n_classes = 30
        ti_min, ti_max = 1.0, 20.0
        ti_step = (ti_max - ti_min) / n_classes
        ti_values = []
        area_fracs = []
        total = 0.0
        for i in range(n_classes):
            ti = ti_min + (i + 0.5) * ti_step
            # Approximate log-normal distribution
            frac = math.exp(-0.5 * ((ti - 7.0) / 3.0) ** 2)
            ti_values.append(ti)
            area_fracs.append(frac)
            total += frac
        area_fracs = [f / total for f in area_fracs]

        ti_lines = "\n".join(
            f"{ti:.4f} {frac:.6f}" for ti, frac in zip(ti_values, area_fracs)
        )

        config_text = f"""szm={params['szm']}
t0={params['t0']}
td={params['td']}
srmax={params['srmax']}
sr0={params['sr0']}
chv={params['chv']}
Q0={params['Q0']}
num_topodex_values={n_classes}
{ti_lines}
"""

        config_file = self.setup_dir / "TOPMODEL" / f"cat-{catchment_id}_topmodel_config.txt"
        config_file.parent.mkdir(parents=True, exist_ok=True)
        with open(config_file, 'w', encoding='utf-8') as f:
            f.write(config_text)

        return config_file

    def generate_sacsma_config(
        self,
        catchment_id: str,
        catchment_row: Optional[gpd.GeoSeries] = None,
        **overrides
    ) -> Path:
        """Generate SAC-SMA BMI configuration files (Fortran namelist format).

        Args:
            catchment_id: Catchment identifier
            catchment_row: Optional GeoSeries with catchment data
            **overrides: Parameter overrides

        Returns:
            Path to generated SAC-SMA control namelist
        """
        hru_area = 100.0  # km^2
        if catchment_row is not None:
            for area_attr in ['area_km2', 'area_sq_km', 'area_km²', 'area', 'catch_area']:
                if area_attr in catchment_row.index and pd.notna(catchment_row[area_attr]):
                    area_val = float(catchment_row[area_attr])
                    hru_area = area_val / 1e6 if area_val > 1000 else area_val
                    break

        # SAC-SMA parameter file entries expected by read_sac_parameters().
        params = {
            # Upper zone
            'uztwm': 50.0,    # Upper zone tension water max [mm]
            'uzfwm': 40.0,    # Upper zone free water max [mm]
            'uzk': 0.3,       # Upper zone lateral depletion [1/day]
            # Lower zone
            'lztwm': 130.0,   # Lower zone tension water max [mm]
            'lzfpm': 50.0,    # Lower zone primary free water max [mm]
            'lzfsm': 25.0,    # Lower zone supplemental free water max [mm]
            'lzpk': 0.01,     # Primary baseflow depletion [1/day]
            'lzsk': 0.05,     # Supplemental baseflow depletion [1/day]
            # Percolation
            'zperc': 40.0,    # Max percolation rate scaling [-]
            'rexp': 2.0,      # Percolation curve exponent [-]
            'pfree': 0.3,     # Fraction percolation to free water [-]
            # Area fractions
            'pctim': 0.01,    # Permanent impervious fraction [-]
            'adimp': 0.1,     # Additional impervious fraction [-]
            'riva': 0.0,      # Riparian vegetation fraction [-]
            'side': 0.0,      # Deep recharge fraction [-]
            'rserv': 0.3,     # Lower zone free water reserve fraction [-]
        }
        # Allow both lower/upper-case override keys for compatibility.
        for key in list(params.keys()):
            if key in overrides:
                params[key] = overrides[key]
            elif key.upper() in overrides:
                params[key] = overrides[key.upper()]

        param_file = self.setup_dir / "SACSMA" / f"cat-{catchment_id}_sacsma_parameters.txt"
        param_file.parent.mkdir(parents=True, exist_ok=True)

        param_lines = [
            f"hru_id cat-{catchment_id}",
            f"hru_area {hru_area:.4f}",
            f"uztwm {params['uztwm']}",
            f"uzfwm {params['uzfwm']}",
            f"lztwm {params['lztwm']}",
            f"lzfpm {params['lzfpm']}",
            f"lzfsm {params['lzfsm']}",
            f"adimp {params['adimp']}",
            f"uzk {params['uzk']}",
            f"lzpk {params['lzpk']}",
            f"lzsk {params['lzsk']}",
            f"zperc {params['zperc']}",
            f"rexp {params['rexp']}",
            f"pctim {params['pctim']}",
            f"pfree {params['pfree']}",
            f"riva {params['riva']}",
            f"side {params['side']}",
            f"rserv {params['rserv']}",
        ]
        with open(param_file, 'w', encoding='utf-8') as f:
            f.write("\n".join(param_lines) + "\n")

        control_params = {
            'main_id': f"cat-{catchment_id}",
            'n_hrus': 1,
            'forcing_root': '',
            'output_root': '',
            'sac_param_file': str(param_file.resolve()),
            'output_hrus': 1,
            'start_datehr': '2010010100',
            'end_datehr': '2020123123',
            'model_timestep': 3600,
            'warm_start_run': 0,
            'write_states': 0,
            'sac_state_in_root': '',
            'sac_state_out_root': '',
        }
        for key, value in overrides.items():
            if key in control_params:
                control_params[key] = value

        config_file = self.setup_dir / "SACSMA" / f"cat-{catchment_id}_sacsma_config.txt"
        control_lines = [
            "! SAC-SMA Control Configuration - Generated by SYMFLUENCE",
            "! Format: Fortran NAMELIST for SAC_CONTROL group",
            "",
            "&SAC_CONTROL",
        ]
        for param_name, param_value in control_params.items():
            if isinstance(param_value, str):
                control_lines.append(f"  {param_name} = '{param_value}',")
            else:
                control_lines.append(f"  {param_name} = {param_value},")
        control_lines.append("/")
        control_lines.append("")

        with open(config_file, 'w', encoding='utf-8') as f:
            f.write("\n".join(control_lines))

        self.logger.info(
            f"Generated SAC-SMA control namelist: {config_file.name}\n"
            f"  Parameter file: {param_file}"
        )

        return config_file

    def generate_snow17_config(
        self,
        catchment_id: str,
        catchment_row: Optional[gpd.GeoSeries] = None,
        **overrides
    ) -> Path:
        """Generate Snow-17 BMI configuration files (Fortran namelist format).

        IMPORTANT: Snow-17's Fortran BMI wrapper requires TWO separate configuration files:
        
        1. **Control Namelist File** (cat-{id}_snow17_config.txt)
           - Fortran NAMELIST format: &SNOW17_CONTROL ... /
           - Contains control parameters like timestep, file paths, dates
           - Loaded by namelistModule.f90::readNamelist() via unit 33
           - This is the PRIMARY config file referenced in NGEN's realization JSON
        
        2. **Parameter File** (cat-{id}_snow17_parameters.txt)
              - Plain text key value format (hru_id, latitude, elevation, SCF, PXTEMP, etc.)
           - Contains snowpack parameters and HRU metadata
           - Path referenced via snow17_param_file in the control namelist
           - Multiple HRU support: one row per HRU
        
        Backwards Compatibility:
        - Detects old format (plain key=value in config file)
        - Migrates to new format (strict Fortran namelists) on next write
        - Logs warning if old format detected
        
        Args:
            catchment_id: Catchment identifier (becomes HRU ID)
            catchment_row: Optional GeoSeries with catchment data (elevation_m, area, etc.)
            **overrides: Parameter overrides for snowpack params (SCF, PXTEMP, MFMAX, etc.)
                        or control params (model_timestep, warm_start_run, etc.)

        Returns:
            Path to generated CONTROL namelist file
            
        Raises:
            ValueError: If required catchment data is missing or invalid
            
        Note:
            Snow-17 model timestep (model_timestep in namelist) should match
            the forcing data timestep (typically 3600 seconds for hourly data).
            See NGEN documentation for temporal alignment requirements.
        """
        # Extract latitude, elevation, and area from catchment
        lat = 45.0
        elevation = 1000.0
        hru_area = 100.0  # Default 100 km² if not provided
        
        if catchment_row is not None:
            centroid = self._get_wgs84_centroid(catchment_row)
            lat = centroid.y
            
            # Elevation: try multiple common attribute names
            for elev_attr in ['elevation_m', 'elevation', 'elev_mean', 'mean_elev', 'elev_m']:
                if elev_attr in catchment_row.index and pd.notna(catchment_row[elev_attr]):
                    elevation = float(catchment_row[elev_attr])
                    break
            
            # Area: try multiple common attribute names (convert to km²)
            for area_attr in ['area_km2', 'area_sq_km', 'area_km²', 'area', 'catch_area']:
                if area_attr in catchment_row.index and pd.notna(catchment_row[area_attr]):
                    area_val = float(catchment_row[area_attr])
                    # Convert from m² to km² if needed
                    if area_val > 1000:  # likely in m²
                        hru_area = area_val / 1e6
                    else:  # likely already in km²
                        hru_area = area_val
                    break

        # =====================================================================
        # SNOWPACK PARAMETERS (per-HRU)
        # =====================================================================
        # Snow-17 Fortran parser expects lowercase labels with whitespace-delimited
        # values and exactly 26 entries including adc1..adc11.
        snowpack_params = {
            'scf': 1.0,       # Snowfall correction factor [-]
            'pxtemp': 1.0,    # Rain/snow threshold [°C]
            'mfmax': 1.0,     # Max melt factor [mm/°C/6hr]
            'mfmin': 0.2,     # Min melt factor [mm/°C/6hr]
            'nmf': 0.15,      # Negative melt factor [mm/°C/6hr]
            'mbase': 0.0,     # Base melt temperature [°C]
            'tipm': 0.1,      # Antecedent temperature index weight [-]
            'uadj': 0.04,     # Rain-on-snow wind function [mm/mb/6hr]
            'plwhc': 0.04,    # Liquid water holding capacity [-]
            'daygm': 0.0,     # Daily ground melt [mm/day]
            'si': 0.0,        # Initial snowpack water equivalent [mm]
            # Areal depletion curve points (required by parser)
            'adc1': 0.0,
            'adc2': 0.1,
            'adc3': 0.2,
            'adc4': 0.3,
            'adc5': 0.4,
            'adc6': 0.5,
            'adc7': 0.6,
            'adc8': 0.7,
            'adc9': 0.8,
            'adc10': 0.9,
            'adc11': 1.0,
        }
        # Allow user overrides for snowpack parameters (accept upper/lower case keys)
        for key in list(snowpack_params.keys()):
            if key in overrides:
                snowpack_params[key] = overrides.pop(key)
            elif key.upper() in overrides:
                snowpack_params[key] = overrides.pop(key.upper())

        # =====================================================================
        # CONTROL PARAMETERS (Fortran NAMELIST format)
        # =====================================================================
        # These control the model execution
        control_params = {
            'main_id': f"cat-{catchment_id}",
            'n_hrus': 1,  # Single HRU per catchment
            'output_hrus': 1,  # Output individual HRU results (1=yes)
            'model_timestep': 3600,  # Seconds; must match forcing data
            'warm_start_run': 0,  # Cold start (0=no warm start)
            'write_states': 0,  # Don't write state files for cold start
            'forcing_root': '',  # Will be set by NGEN at runtime
            'output_root': '',  # Will be set by NGEN at runtime
            'snow_state_in_root': '',  # Optional: warm-start state file path
            'snow_state_out_root': '',  # Optional: output state file path
            'snow17_param_file': f'cat-{catchment_id}_snow17_parameters.txt',  # Normalized to absolute path below
            'start_datehr': '2010010100',  # Default; will be overridden at runtime
            'end_datehr': '2020123123',    # Default; will be overridden at runtime
        }
        # Allow user overrides for control parameters
        control_params.update({k: v for k, v in overrides.items() if k in control_params})

        # =====================================================================
        # WRITE PARAMETER FILE (cat-{id}_snow17_parameters.txt)
        # =====================================================================
        # Strict format expected by snow17/src/share/ioModule.f90:
        #   <param_name><whitespace><value>
        # No comment lines and no key=value syntax.
        param_lines = [
            f"hru_id cat-{catchment_id}",
            f"hru_area {hru_area:.4f}",
            f"latitude {lat:.4f}",
            f"elev {elevation:.1f}",
            f"mfmax {snowpack_params['mfmax']}",
            f"mfmin {snowpack_params['mfmin']}",
            f"scf {snowpack_params['scf']}",
            f"uadj {snowpack_params['uadj']}",
            f"si {snowpack_params['si']}",
            f"pxtemp {snowpack_params['pxtemp']}",
            f"nmf {snowpack_params['nmf']}",
            f"tipm {snowpack_params['tipm']}",
            f"mbase {snowpack_params['mbase']}",
            f"plwhc {snowpack_params['plwhc']}",
            f"daygm {snowpack_params['daygm']}",
            f"adc1 {snowpack_params['adc1']}",
            f"adc2 {snowpack_params['adc2']}",
            f"adc3 {snowpack_params['adc3']}",
            f"adc4 {snowpack_params['adc4']}",
            f"adc5 {snowpack_params['adc5']}",
            f"adc6 {snowpack_params['adc6']}",
            f"adc7 {snowpack_params['adc7']}",
            f"adc8 {snowpack_params['adc8']}",
            f"adc9 {snowpack_params['adc9']}",
            f"adc10 {snowpack_params['adc10']}",
            f"adc11 {snowpack_params['adc11']}",
        ]

        param_file = self.setup_dir / "SNOW17" / f"cat-{catchment_id}_snow17_parameters.txt"
        param_file.parent.mkdir(parents=True, exist_ok=True)
        with open(param_file, 'w', encoding='utf-8') as f:
            f.write("\n".join(param_lines) + "\n")

        # Fortran SNOW17 resolves this path relative to NGEN's process CWD, not init_config location.
        # Normalize to an absolute path to avoid runtime "Cannot open file ..." failures.
        configured_param_path = str(control_params.get('snow17_param_file', '')).strip()
        default_param_name = f'cat-{catchment_id}_snow17_parameters.txt'
        if not configured_param_path or configured_param_path == default_param_name:
            resolved_param_file = param_file.resolve()
        else:
            candidate = Path(configured_param_path).expanduser()
            if not candidate.is_absolute():
                candidate = (param_file.parent / candidate)
            resolved_param_file = candidate.resolve()
        control_params['snow17_param_file'] = str(resolved_param_file)

        self.logger.debug(
            f"Generated Snow-17 parameter file: {param_file.name}\n"
            f"  HRU ID: {catchment_id}\n"
            f"  Latitude: {lat:.4f}°, Elevation: {elevation:.1f}m, Area: {hru_area:.4f}km²"
        )

        # =====================================================================
        # WRITE CONTROL NAMELIST FILE (cat-{id}_snow17_config.txt)
        # =====================================================================
        # Fortran NAMELIST format: &SECTION_NAME ... /
        # This is read by Snow-17's namelistModule.f90::readNamelist()
        control_lines = [
            "! Snow-17 Control Configuration - Generated by SYMFLUENCE",
            "! Format: Fortran NAMELIST for SNOW17_CONTROL group",
            "! Loaded by namelistModule.f90::readNamelist() via UNIT 33",
            "! See Snow-17 source code for parameter definitions",
            "",
            "&SNOW17_CONTROL",
        ]
        # Write control parameters in namelist format
        for param_name, param_value in control_params.items():
            if isinstance(param_value, str):
                # String values in quotes
                control_lines.append(f"  {param_name} = '{param_value}',")
            elif isinstance(param_value, bool):
                # Fortran logical values
                control_lines.append(f"  {param_name} = {1 if param_value else 0},")
            else:
                # Numeric values
                control_lines.append(f"  {param_name} = {param_value},")
        
        control_lines.append("/")  # Namelist terminator
        control_lines.append("")   # Blank line for readability

        config_file = self.setup_dir / "SNOW17" / f"cat-{catchment_id}_snow17_config.txt"

        # Warn when an existing legacy single-file format is present before overwrite.
        self.detect_and_migrate_snow17_format(config_file)

        with open(config_file, 'w', encoding='utf-8') as f:
            f.write("\n".join(control_lines))

        self.logger.info(
            f"Generated Snow-17 control namelist: {config_file.name}\n"
            f"  Format: Fortran NAMELIST (&SNOW17_CONTROL ... /)\n"
            f"  Main ID: {control_params['main_id']}\n"
            f"  Parameter file: {control_params['snow17_param_file']}"
        )

        return config_file

    def detect_and_migrate_snow17_format(self, config_file: Path) -> bool:
        """
        Detect and migrate old-format Snow-17 configs to new Fortran namelist format.

        **BACKWARDS COMPATIBILITY HELPER**

        Context:
        - Old format: Plain key=value file mixing snowpack params and (non-existent) control params
        - New format: Proper Fortran NAMELIST (&SNOW17_CONTROL ... /) with separate parameter file
        - Snow-17's Fortran code REQUIRES the new format from namelistModule.f90

        Detection:
        - Old format: Lacks "&SNOW17_CONTROL" namelist marker
        - New format: Starts with "&SNOW17_CONTROL"

        Args:
            config_file: Path to Snow-17 config file

        Returns:
            True if old format detected (needs migration)
            False if new format (OK)

        Side Effects:
        - Logs warning if old format is detected
        - Recommends re-running preprocessing to auto-generate new format
        """
        if not config_file.exists():
            return False

        try:
            content = config_file.read_text(encoding='utf-8')
            is_old_format = '&SNOW17_CONTROL' not in content and 'SCF=' in content
            
            if is_old_format:
                self.logger.warning(
                    f"DEPRECATED: Old Snow-17 config format detected at {config_file.name}\n"
                    f"  Current format: Plain key=value (non-functional)\n"
                    f"  Required format: Fortran NAMELIST (&SNOW17_CONTROL ... /)\n"
                    f"  Solution: Re-run NGEN preprocessing to auto-migrate config files.\n"
                    f"  The next `sf workflow run` will regenerate configs in the correct format.\n"
                    f"  No manual action needed; this warning can be safely ignored."
                )
                return True
            return False
        except Exception as e:
            self.logger.debug(f"Could not detect Snow-17 config format: {e}")
            return False

    def generate_realization_config(
        self,
        forcing_file: Path,
        project_dir: Path,
        lib_paths: Optional[Dict[str, Path]] = None
    ) -> Path:
        """
        Generate ngen realization configuration JSON.

        Args:
            forcing_file: Path to forcing NetCDF file
            project_dir: Project directory for output paths
            lib_paths: Optional dict of module library paths

        Returns:
            Path to generated realization config
        """
        self.logger.info("Generating realization configuration")

        forcing_abs_path = str(forcing_file.resolve())
        cfe_config_base = str((self.setup_dir / "CFE").resolve())
        pet_config_base = str((self.setup_dir / "PET").resolve())
        noah_config_base = str((self.setup_dir / "NOAH").resolve())
        topmodel_config_base = str((self.setup_dir / "TOPMODEL").resolve())
        sacsma_config_base = str((self.setup_dir / "SACSMA").resolve())
        snow17_config_base = str((self.setup_dir / "SNOW17").resolve())

        lib_ext = ".dylib" if sys.platform == "darwin" else ".so"

        forcing_provider = self._get_config_value(lambda: self.config.model.ngen.forcing_provider, default=None)
        if not forcing_provider:
            # CsvPerFeature: works but has SIGSEGV bug on macOS ARM64 (~19% crash rate,
            # mitigated by retry logic in runner.py).
            # NetCDF: avoids SIGSEGV but requires AORC-format NetCDF with epoch_start
            # attribute and CSDMS variable names. Our forcing pipeline doesn't yet
            # produce this format, so default to CsvPerFeature for now.
            forcing_provider = "CsvPerFeature"

        # Handle simulation times
        sim_start = self._get_config_value(lambda: self.config.domain.time_start, default='2000-01-01 00:00:00', dict_key='EXPERIMENT_TIME_START')
        sim_end = self._get_config_value(lambda: self.config.domain.time_end, default='2000-12-31 23:00:00', dict_key='EXPERIMENT_TIME_END')

        if sim_start == 'default':
            sim_start = '2000-01-01 00:00:00'
        if sim_end == 'default':
            sim_end = '2000-12-31 23:00:00'

        sim_start = pd.to_datetime(sim_start).strftime('%Y-%m-%d %H:%M:%S')
        sim_end = pd.to_datetime(sim_end).strftime('%Y-%m-%d %H:%M:%S')

        # Get forcing timestep from config (default to 3600 if not specified)
        forcing_timestep = self._get_config_value(lambda: self.config.forcing.time_step_size, default=3600, dict_key='FORCING_TIME_STEP_SIZE')
        try:
            forcing_timestep = int(forcing_timestep)
            self.logger.info(f"Using forcing timestep: {forcing_timestep} seconds ({forcing_timestep/3600:.1f} hours)")
        except (ValueError, TypeError):
            self.logger.warning(f"Invalid FORCING_TIME_STEP_SIZE: {forcing_timestep}, using default 3600")
            forcing_timestep = 3600

        # Build forcing config
        if forcing_provider == "CsvPerFeature":
            forcing_config = {
                "path": str((forcing_file.parent / "csv").resolve()),
                "provider": "CsvPerFeature",
                "file_pattern": ".*{{id}}_forcing.*\\.csv"
            }
        else:
            forcing_config = {
                "path": forcing_abs_path,
                "provider": forcing_provider
            }

        # Build module configurations
        modules = self._build_module_configs(
            cfe_config_base, pet_config_base, noah_config_base, lib_ext,
            lib_paths=lib_paths,
            topmodel_base=topmodel_config_base,
            sacsma_base=sacsma_config_base,
            snow17_base=snow17_config_base,
        )

        experiment_id = self._get_config_value(lambda: self.config.domain.experiment_id, default='default_run', dict_key='EXPERIMENT_ID')
        output_root = str((project_dir / "simulations" / experiment_id / "NGEN").resolve())

        # Determine model_type_name and main_output_variable based on active modules
        if self._include_topmodel:
            model_type = "bmi_multi_topmodel"
            main_output = "Qout"
        elif self._include_sacsma:
            model_type = "bmi_multi_sacsma"
            main_output = "tci"
        else:
            model_type = "bmi_multi_noahowp_cfe"
            main_output = "Q_OUT"

        # Prepend land-surface/snow prefix
        if self._include_noah:
            model_type = f"bmi_multi_noahowp_{model_type.split('_', 3)[-1]}"
        elif self._include_snow17:
            model_type = f"bmi_multi_snow17_{model_type.split('_', 3)[-1]}"

        config = {
            "global": {
                "formulations": [{
                    "name": "bmi_multi",
                    "params": {
                        "model_type_name": model_type,
                        "init_config": "",
                        "allow_exceed_end_time": True,
                        "main_output_variable": main_output,
                        "modules": modules,
                    }
                }],
                "forcing": forcing_config
            },
            "time": {
                "start_time": sim_start,
                "end_time": sim_end,
                "output_interval": forcing_timestep
            },
            "output_root": output_root
        }

        config_file = self.setup_dir / "realization_config.json"
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)

        self.logger.info(f"Created realization config: {config_file}")
        return config_file

    def _build_runoff_variables_map(self) -> dict:
        """Build the variables_names_map for the active runoff module.

        Determines precipitation and ET sources based on which upstream
        modules are enabled:
        - Precip: NOAH QINSUR > Snow-17 rain_plus_melt > raw forcing
        - ET: PET > NOAH fallback
        """
        variables_map = {}

        # Precipitation source
        if self._include_noah:
            variables_map["atmosphere_water__liquid_equivalent_precipitation_rate"] = "QINSUR"
        elif self._include_snow17:
            variables_map["atmosphere_water__liquid_equivalent_precipitation_rate"] = "rain_plus_melt"
        else:
            variables_map["atmosphere_water__liquid_equivalent_precipitation_rate"] = (
                "atmosphere_water__liquid_equivalent_precipitation_rate"
            )

        # ET source
        if self._include_pet:
            variables_map["water_potential_evaporation_flux"] = "water_potential_evaporation_flux"
        elif self._include_noah:
            et_var = getattr(self, '_noah_et_fallback', 'EVAPOTRANS')
            variables_map["water_potential_evaporation_flux"] = et_var

        return variables_map

    def _build_module_configs(
        self,
        cfe_base: str,
        pet_base: str,
        noah_base: str,
        lib_ext: str,
        lib_paths: Optional[Dict[str, Path]] = None,
        topmodel_base: str = "",
        sacsma_base: str = "",
        snow17_base: str = "",
    ) -> list:
        """Build the list of module configurations for realization."""
        modules = []
        lib_paths = lib_paths or {}
        provide_pet_rel_humidity = self._include_pet and not self._include_sloth

        if self._include_sloth or provide_pet_rel_humidity:
            lib_file = str(lib_paths.get("SLOTH", f"./extern/sloth/cmake_build/libslothmodel{lib_ext}"))
            modules.append({
                "name": "bmi_c++",
                "params": {
                    "model_type_name": "bmi_c++_sloth",
                    "library_file": lib_file,
                    "init_config": "/dev/null",
                    "allow_exceed_end_time": True,
                    "main_output_variable": "z",
                    "uses_forcing_file": False,
                    "model_params": {
                        "sloth_ice_fraction_schaake(1,double,m,node)": 0.0,
                        "sloth_ice_fraction_xinanjiang(1,double,1,node)": 0.0,
                        "sloth_smp(1,double,1,node)": 0.0,
                        "sloth_atmosphere_air_water~vapor__relative_saturation(1,double,1,node)": 0.5
                    }
                }
            })

        if self._include_pet:
            lib_file = str(lib_paths.get("PET", f"./extern/evapotranspiration/evapotranspiration/cmake_build/libpetbmi{lib_ext}"))
            pet_params: dict[str, Any] = {
                "model_type_name": "bmi_c_pet",
                "library_file": lib_file,
                "forcing_file": "",
                "init_config": f"{pet_base}/{{{{id}}}}_pet_config.txt",
                "allow_exceed_end_time": True,
                "main_output_variable": "water_potential_evaporation_flux",
                "registration_function": "register_bmi_pet",
                "uses_forcing_file": False
            }
            # PET requires atmosphere_air_water~vapor__relative_saturation as a BMI input.
            # When SLOTH is enabled (explicitly or as an internal PET dependency),
            # it provides this as a dummy value; PET with yes_aorc=1
            # computes actual humidity internally from specific humidity in forcing.
            if self._include_sloth or provide_pet_rel_humidity:
                pet_params["variables_names_map"] = {
                    "atmosphere_air_water~vapor__relative_saturation":
                        "sloth_atmosphere_air_water~vapor__relative_saturation"
                }
            pet_config = {
                "name": "bmi_c",
                "params": pet_params
            }
            modules.append(pet_config)

        if self._include_noah:
            lib_file = str(lib_paths.get("NOAH", f"./extern/noah-owp-modular/cmake_build/libsurfacebmi{lib_ext}"))
            modules.append({
                "name": "bmi_fortran",
                "params": {
                    "model_type_name": "bmi_fortran_noahowp",
                    "library_file": lib_file,
                    "forcing_file": "",
                    "init_config": f"{noah_base}/{{{{id}}}}.input",
                    "allow_exceed_end_time": True,
                    "main_output_variable": "QINSUR",
                    "variables_names_map": {
                        "PRCPNONC": "atmosphere_water__liquid_equivalent_precipitation_rate",
                        "Q2": "atmosphere_air_water~vapor__specific_humidity",
                        "SFCTMP": "land_surface_air__temperature",
                        "UU": "land_surface_wind__x_component_of_velocity",
                        "VV": "land_surface_wind__y_component_of_velocity",
                        "LWDN": "land_surface_radiation~incoming~longwave__energy_flux",
                        "SOLDN": "land_surface_radiation~incoming~shortwave__energy_flux",
                        "SFCPRS": "land_surface_air__pressure"
                    },
                    # Expose NOAH outputs for coupling with CFE
                    # Variable names must match NOAH-OWP BMI output_items
                    # (see bmi_noahowp.f90 noahowp_output_var_names)
                    "output_variables": [
                        "QINSUR",       # Net water input to soil surface [m/s]
                        "EVAPOTRANS",   # Total evapotranspiration rate [m/s]
                        "ETRAN",        # Transpiration [mm] (accumulated per timestep)
                        "ECAN",         # Canopy interception evaporation [mm]
                        "QSEVA"         # Direct soil evaporation rate [mm/s]
                    ]
                }
            })

        if self._include_snow17:
            lib_file = str(lib_paths.get("SNOW17", f"./extern/snow17/cmake_build/libsnow17_bmi{lib_ext}"))
            snow17_vars = {
                "tair": "land_surface_air__temperature",
                "precip": "atmosphere_water__liquid_equivalent_precipitation_rate",
                "raim": "rain_plus_melt",
            }
            modules.append({
                "name": "bmi_fortran",
                "params": {
                    "model_type_name": "bmi_fortran_snow17",
                    "library_file": lib_file,
                    "forcing_file": "",
                    "init_config": f"{snow17_base}/{{{{id}}}}_snow17_config.txt",
                    "allow_exceed_end_time": True,
                    "main_output_variable": "raim",
                    "variables_names_map": snow17_vars,
                    "output_variables": ["raim", "sneqv"]
                }
            })

        if self._include_cfe:
            lib_file = str(lib_paths.get("CFE", f"./extern/cfe/cmake_build/libcfebmi{lib_ext}"))
            variables_map = self._build_runoff_variables_map()

            # Add SLOTH variables if SLOTH is enabled (provides ice fraction for partitioning)
            if self._include_sloth:
                variables_map["ice_fraction_schaake"] = "sloth_ice_fraction_schaake"
                variables_map["ice_fraction_xinanjiang"] = "sloth_ice_fraction_xinanjiang"
                variables_map["soil_moisture_profile"] = "sloth_smp"

            modules.append({
                "name": "bmi_c",
                "params": {
                    "model_type_name": "bmi_c_cfe",
                    "library_file": lib_file,
                    "forcing_file": "",
                    "init_config": f"{cfe_base}/{{{{id}}}}_bmi_config_cfe_pass.txt",
                    "allow_exceed_end_time": True,
                    "main_output_variable": "Q_OUT",
                    "registration_function": "register_bmi_cfe",
                    "variables_names_map": variables_map,
                    "output_variable_units": "m3/s"
                }
            })

        if self._include_topmodel:
            lib_file = str(lib_paths.get("TOPMODEL", f"./extern/topmodel/cmake_build/libtopmodelbmi{lib_ext}"))
            variables_map = self._build_runoff_variables_map()

            modules.append({
                "name": "bmi_c",
                "params": {
                    "model_type_name": "bmi_c_topmodel",
                    "library_file": lib_file,
                    "forcing_file": "",
                    "init_config": f"{topmodel_base}/{{{{id}}}}_topmodel_config.txt",
                    "allow_exceed_end_time": True,
                    "main_output_variable": "Qout",
                    "registration_function": "register_bmi_topmodel",
                    "variables_names_map": variables_map,
                    "output_variable_units": "m3/s"
                }
            })

        if self._include_sacsma:
            lib_file = str(lib_paths.get("SACSMA", f"./extern/sac-sma/cmake_build/libsacbmi{lib_ext}"))
            variables_map = self._build_runoff_variables_map()
            sacsma_variables_map = dict(variables_map)

            # SAC-SMA BMI expects raw input names (tair, precip, pet).
            # Provide explicit aliases to avoid deferred provision failures.
            if self._include_noah:
                precip_source = "QINSUR"
            elif self._include_snow17:
                precip_source = "rain_plus_melt"
            else:
                precip_source = "atmosphere_water__liquid_equivalent_precipitation_rate"

            if self._include_pet:
                pet_source = "water_potential_evaporation_flux"
            elif self._include_noah:
                pet_source = getattr(self, '_noah_et_fallback', 'EVAPOTRANS')
            else:
                pet_source = None

            sacsma_variables_map["tair"] = "land_surface_air__temperature"
            sacsma_variables_map["precip"] = precip_source
            if pet_source:
                sacsma_variables_map["pet"] = pet_source

            modules.append({
                "name": "bmi_fortran",
                "params": {
                    "model_type_name": "bmi_fortran_sacsma",
                    "library_file": lib_file,
                    "forcing_file": "",
                    "init_config": f"{sacsma_base}/{{{{id}}}}_sacsma_config.txt",
                    "allow_exceed_end_time": True,
                    "main_output_variable": "tci",
                    "variables_names_map": sacsma_variables_map,
                    "output_variable_units": "m3/s"
                }
            })

        return modules

    @staticmethod
    def _generate_giuh_ordinates(catchment_area_km2: float) -> str:
        """Generate GIUH ordinates appropriate for basin size.

        Uses a gamma-distribution-shaped unit hydrograph with response time
        scaled to catchment area. Small basins get short, peaked responses;
        large basins get longer, flatter responses.

        Args:
            catchment_area_km2: Catchment area in km²

        Returns:
            Comma-separated string of ordinates summing to 1.0
        """
        import math

        # Estimate response time in hours (empirical: sqrt(area) / 2, min 2h)
        response_time = max(math.sqrt(catchment_area_km2) / 2.0, 2.0)

        # Number of ordinates: cover ~2x response time, min 5
        n_ordinates = max(int(response_time * 2), 5)
        # Cap at 48 ordinates (2 days at hourly timestep)
        n_ordinates = min(n_ordinates, 48)

        # Gamma distribution shape: k=2 gives realistic geomorphological response
        k = 2.0
        theta = response_time / k  # scale parameter

        # Generate gamma PDF values at each hour
        ordinates = []
        for i in range(1, n_ordinates + 1):
            t = float(i)
            # Gamma PDF: t^(k-1) * exp(-t/theta) / (theta^k * Gamma(k))
            val = (t ** (k - 1)) * math.exp(-t / theta) / (theta ** k * math.gamma(k))
            ordinates.append(val)

        # Normalize to sum to 1.0
        total = sum(ordinates)
        if total > 0:
            ordinates = [o / total for o in ordinates]

        # Format with 4 decimal places
        return ','.join(f'{o:.4f}' for o in ordinates)

    def _get_wgs84_centroid(self, catchment_row: gpd.GeoSeries):
        """Get the centroid of a catchment in WGS84 coordinates."""
        centroid = catchment_row.geometry.centroid

        if self.catchment_crs and str(self.catchment_crs) != "EPSG:4326":
            geom_wgs84 = gpd.GeoSeries([catchment_row.geometry], crs=self.catchment_crs)
            geom_wgs84 = geom_wgs84.to_crs("EPSG:4326")
            centroid = geom_wgs84.iloc[0].centroid

        return centroid

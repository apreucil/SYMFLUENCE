# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
NGEN Forcing Adjuster

Applies forcing adjustment parameters (PXADJ) to forcing data
for calibration. Adjustments are applied in-memory to cached base forcing
before writing iteration-specific forcing files.

Architecture:
    - Loads cached base forcing (from preprocessor)
    - Applies multiplicative adjustments (PXADJ * precipitation)
    - Writes adjusted forcing to worker-specific directories
    - Updates realization.json to point to adjusted forcing
    
Usage:
    from symfluence.models.ngen.calibration.forcing_adjuster import (
        NgenForcingAdjuster, apply_forcing_adjustments
    )
    
    # Convenience function
    adjusted_path = apply_forcing_adjustments(params, settings_dir, config, logger)
    
    # Or use class directly
    adjuster = NgenForcingAdjuster(config, logger)
    forcing = adjuster.load_base_forcing()
    adjusted, log = adjuster.apply_adjustments(forcing, params)
    adjuster.write_adjusted_forcing(adjusted, settings_dir)
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr


class NgenForcingAdjuster:
    """
    Applies forcing adjustments for NGEN calibration.
    
    Handles:
    - Loading cached base forcing data
    - Applying PXADJ (precipitation adjustment)
    - Writing adjusted forcing to iteration-specific directories
    - Updating CSV files for consistency
    - Patching realization config to point to adjusted forcing
    
    Thread-safety: This class is thread-safe for read operations but not
    for write operations. Each worker process should have its own instance.
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        logger: logging.Logger,
        base_forcing_cache: Optional[xr.Dataset] = None
    ):
        """
        Initialize forcing adjuster.
        
        Args:
            config: Configuration dictionary
            logger: Logger instance
            base_forcing_cache: Optional pre-loaded base forcing (for performance)
        """
        self.config = config
        self.logger = logger
        self._base_forcing_cache = base_forcing_cache
        
        # Get project paths
        data_dir = Path(config.get('SYMFLUENCE_DATA_DIR', '.'))
        domain_name = config.get('DOMAIN_NAME', '')
        self.project_dir = data_dir / f"domain_{domain_name}"
        self.cache_dir = self.project_dir / "cache"
        
        # Cache paths
        self.base_forcing_path = self.cache_dir / "forcing_base.nc"
        self.cache_metadata_path = self.cache_dir / "forcing_cache_metadata.json"
        
        # Load cache metadata
        self._load_cache_metadata()
    
    def _load_cache_metadata(self):
        """Load forcing cache metadata."""
        if not self.cache_metadata_path.exists():
            self.logger.warning(
                f"Forcing cache metadata not found: {self.cache_metadata_path}"
            )
            self.catchment_ids = []
            self.metadata = {}
            return
        
        try:
            with open(self.cache_metadata_path) as f:
                self.metadata = json.load(f)
                self.catchment_ids = self.metadata.get('catchment_ids', [])
                self.logger.debug(
                    f"Loaded cache metadata: {len(self.catchment_ids)} catchments, "
                    f"{self.metadata.get('n_timesteps', 0)} timesteps"
                )
        except Exception as e:
            self.logger.error(f"Error loading cache metadata: {e}")
            self.catchment_ids = []
            self.metadata = {}
    
    def load_base_forcing(self) -> xr.Dataset:
        """
        Load cached base forcing data.
        
        Uses in-memory cache if available (for performance).
        
        Returns:
            xarray Dataset with base forcing
            
        Raises:
            FileNotFoundError: If base forcing cache doesn't exist
        """
        # Check in-memory cache first
        if self._base_forcing_cache is not None:
            self.logger.debug("Using in-memory cached base forcing")
            return self._base_forcing_cache.copy(deep=True)
        
        if not self.base_forcing_path.exists():
            raise FileNotFoundError(
                f"Base forcing cache not found: {self.base_forcing_path}. "
                f"Run preprocessing first to generate cache."
            )
        
        self.logger.debug(f"Loading base forcing from {self.base_forcing_path}")
        forcing = xr.open_dataset(self.base_forcing_path)
        
        # Cache for future calls
        self._base_forcing_cache = forcing
        
        return forcing.copy(deep=True)
    
    def detect_precip_variable(
        self,
        forcing_data: xr.Dataset
    ) -> Optional[str]:
        """
        Detect which precipitation variable to adjust based on available variables.
        
        Tries common precipitation variable names in priority order.
        
        Args:
            forcing_data: Forcing dataset
            
        Returns:
            Variable name to adjust, or None if no precipitation variable found
        """
        # Priority order for precipitation variable names
        precip_candidates = [
            'APCP_surface',  # AORC format
            'precip_rate',   # NGEN CSV format
            'precipitation', # Generic
            'RAINRATE',      # Some datasets
            'pcp',           # Abbreviation
            'PRECIP',        # Uppercase
        ]
        
        for var in precip_candidates:
            if var in forcing_data:
                self.logger.debug(f"Using precipitation variable '{var}' for PXADJ adjustment")
                return var
        
        self.logger.error(
            "No precipitation variable found in forcing data. PXADJ adjustment will be skipped. "
            f"Available variables: {', '.join(list(forcing_data.data_vars.keys()))}"
        )
        return None
    
    def apply_adjustments(
        self,
        forcing_data: xr.Dataset,
        adjustments: Dict[str, float]
    ) -> Tuple[xr.Dataset, Dict[str, str]]:
        """
        Apply forcing adjustments to dataset.
        
        Multiplicative adjustments:
        - FORCING.PXADJ: Multiply precipitation by adjustment factor
        
        Args:
            forcing_data: Base forcing dataset
            adjustments: Dict with FORCING.PXADJ value
            
        Returns:
            Tuple of (adjusted_dataset, adjustment_log)
            
        Example:
            >>> forcing = adjuster.load_base_forcing()
            >>> params = {'FORCING.PXADJ': 1.15}
            >>> adjusted, log = adjuster.apply_adjustments(forcing, params)
            >>> print(log['PXADJ'])
            'Applied PXADJ=1.150 to precip_rate. Mean changed: 2.5 → 2.875'
        """
        adjusted = forcing_data.copy(deep=True)
        adjustment_log = {}
        
        # Apply PXADJ (precipitation adjustment)
        if 'FORCING.PXADJ' in adjustments:
            pxadj = adjustments['FORCING.PXADJ']
            
            # Validate bounds
            if not (0.1 <= pxadj <= 3.0):
                self.logger.warning(
                    f"PXADJ={pxadj:.3f} is outside typical range [0.1, 3.0]. "
                    f"This may indicate an issue with parameter bounds."
                )
            
            precip_var = self.detect_precip_variable(adjusted)
            
            if precip_var:
                original_mean = float(adjusted[precip_var].mean())
                original_total = float(adjusted[precip_var].sum())
                
                adjusted[precip_var] = adjusted[precip_var] * pxadj
                
                adjusted_mean = float(adjusted[precip_var].mean())
                adjusted_total = float(adjusted[precip_var].sum())
                
                adjustment_log['PXADJ'] = (
                    f"Applied PXADJ={pxadj:.3f} to {precip_var}. "
                    f"Mean: {original_mean:.4f} → {adjusted_mean:.4f} "
                    f"(Total: {original_total:.1f} → {adjusted_total:.1f})"
                )
                self.logger.debug(adjustment_log['PXADJ'])
            else:
                warning = "PXADJ specified but no precipitation variable found in forcing data"
                self.logger.warning(warning)
                adjustment_log['PXADJ'] = f"SKIPPED: {warning}"
        
        # Add adjustment metadata to dataset attributes
        # Note: NetCDF4 doesn't support boolean attributes, use int (0 or 1)
        adjusted.attrs['forcing_adjustments_applied'] = 1
        adjusted.attrs['pxadj_value'] = float(adjustments.get('FORCING.PXADJ', 1.0))
        
        return adjusted, adjustment_log
    
    def write_adjusted_forcing(
        self,
        forcing_data: xr.Dataset,
        output_dir: Path
    ) -> Path:
        """
        Write adjusted forcing to NetCDF and CSV files.
        
        Creates:
        - output_dir/NGEN/forcing_adjusted.nc (NetCDF format)
        - output_dir/NGEN/csv/{catchment}_forcing.csv (CSV format, per-catchment)
        
        Args:
            forcing_data: Adjusted forcing dataset
            output_dir: Output directory (e.g., settings_worker_0/ or settings_worker_0/NGEN/)
            
        Returns:
            Path to written NetCDF file
        """
        # Ensure we're in NGEN subdirectory
        ngen_dir = output_dir / "NGEN" if output_dir.name != "NGEN" else output_dir
        ngen_dir.mkdir(parents=True, exist_ok=True)
        
        # Write NetCDF
        netcdf_path = ngen_dir / "forcing_adjusted.nc"
        self.logger.debug(f"Writing adjusted forcing NetCDF to {netcdf_path}")
        forcing_data.to_netcdf(netcdf_path, format='NETCDF4')
        
        # Write CSV files (required for some NGEN module configs)
        csv_dir = ngen_dir / "csv"
        csv_dir.mkdir(parents=True, exist_ok=True)
        
        self._write_csv_forcing(forcing_data, csv_dir)
        
        return netcdf_path
    
    def _write_csv_forcing(self, forcing_data: xr.Dataset, csv_dir: Path):
        """
        Write forcing data to per-catchment CSV files.
        
        Mirrors logic from preprocessor._write_csv_forcing_files()
        """
        # Get catchment dimension
        catchment_dim = None
        for dim in ['catchment-id', 'catchment_id', 'catchment']:
            if dim in forcing_data.dims:
                catchment_dim = dim
                break
        
        if catchment_dim:
            catchments = forcing_data[catchment_dim].values
        else:
            # Single catchment case
            self.logger.warning("No catchment dimension found, writing single CSV")
            catchments = self.catchment_ids or ['cat-1']
        
        for catchment in catchments:
            if catchment_dim:
                catch_data = forcing_data.sel({catchment_dim: catchment})
            else:
                catch_data = forcing_data
            
            # Convert to DataFrame
            df = catch_data.to_dataframe()
            
            # Drop catchment-id column if present (redundant in single-catchment file)
            if catchment_dim in df.columns:
                df = df.drop(columns=[catchment_dim])
            
            # Write CSV
            csv_path = csv_dir / f"{catchment}_forcing.csv"
            df.to_csv(csv_path)
            self.logger.debug(f"Wrote CSV forcing to {csv_path} ({len(df)} timesteps)")
    
    def update_realization_forcing_path(
        self,
        realization_path: Path,
        new_forcing_path: Path
    ):
        """
        Update realization config to point to adjusted forcing file.
        
        Modifies the 'global.forcing.file_pattern' field in realization_config.json
        to point to the adjusted forcing file instead of the default forcing.nc.
        
        Args:
            realization_path: Path to realization_config.json
            new_forcing_path: Path to adjusted forcing file (forcing_adjusted.nc)
        """
        if not realization_path.exists():
            self.logger.warning(f"Realization config not found: {realization_path}")
            return
        
        try:
            with open(realization_path) as f:
                config = json.load(f)
            
            # Update forcing path in global section
            if 'global' in config and 'forcing' in config['global']:
                forcing_config = config['global']['forcing']
                
                # Update file pattern
                if 'file_pattern' in forcing_config:
                    old_pattern = forcing_config['file_pattern']
                    forcing_config['file_pattern'] = str(new_forcing_path)
                    self.logger.debug(
                        f"Updated forcing path in realization: "
                        f"{old_pattern} → {new_forcing_path}"
                    )
                
                # Update path if present
                if 'path' in forcing_config:
                    forcing_config['path'] = str(new_forcing_path.parent)
            
            # Write updated config
            with open(realization_path, 'w') as f:
                json.dump(config, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Error updating realization config: {e}")


def apply_forcing_adjustments(
    params: Dict[str, float],
    settings_dir: Path,
    config: Dict[str, Any],
    logger: logging.Logger
) -> Optional[Path]:
    """
    Convenience function to apply forcing adjustments.
    
    This is the main entry point for forcing adjustment workflow.
    Checks for FORCING.* parameters, applies adjustments if present,
    and updates the realization config.
    
    Args:
        params: Parameter dictionary (may contain FORCING.* params)
        settings_dir: Worker settings directory
        config: Configuration dictionary
        logger: Logger instance
        
    Returns:
        Path to adjusted forcing file, or None if no adjustments needed
        
    Example:
        >>> params = {'SACSMA.UZTWM': 150.0, 'SACSMA.PXADJ': 1.15}
        >>> adjusted_path = apply_forcing_adjustments(params, settings_dir, config, logger)
        >>> if adjusted_path:
        ...     print(f"Using adjusted forcing: {adjusted_path}")
    """
    # Extract forcing adjustment parameters (PXADJ)
    # These may have various module prefixes (SACSMA.PXADJ, FORCING.PXADJ, etc.)
    forcing_param_names = {'PXADJ'}
    forcing_params = {}
    for k, v in params.items():
        # Strip module prefix to get base param name
        param_name = k.split('.')[-1] if '.' in k else k
        if param_name in forcing_param_names:
            # Normalize to FORCING.* format for apply_adjustments
            forcing_params[f'FORCING.{param_name}'] = v
    
    if not forcing_params:
        return None
    
    logger.debug(f"Applying forcing adjustments: {forcing_params}")
    
    try:
        # Create adjuster
        adjuster = NgenForcingAdjuster(config, logger)
        
        # Load base forcing
        base_forcing = adjuster.load_base_forcing()
        
        # Apply adjustments
        adjusted_forcing, adjustment_log = adjuster.apply_adjustments(
            base_forcing, forcing_params
        )
        
        # Write adjusted forcing
        adjusted_path = adjuster.write_adjusted_forcing(adjusted_forcing, settings_dir)
        
        # Update realization config
        realization_path = settings_dir / "NGEN" / "realization_config.json"
        if realization_path.exists():
            adjuster.update_realization_forcing_path(realization_path, adjusted_path)
        
        # Log summary (debug level to reduce verbosity)
        for param, log_msg in adjustment_log.items():
            logger.debug(f"  {param}: {log_msg}")
        
        return adjusted_path
        
    except Exception as e:
        logger.error(f"Error in forcing adjustment workflow: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

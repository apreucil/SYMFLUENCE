"""Tests for NGEN model configuration."""

import pytest

from symfluence.models.base import ConfigValidationError


class TestNgenConfigAdapter:
    """Tests for NgenConfigAdapter."""

    def test_adapter_can_be_imported(self):
        from symfluence.models.ngen.config import NgenConfigAdapter
        assert NgenConfigAdapter is not None

    def test_backward_compat_alias(self):
        from symfluence.models.ngen.config import NGENConfigAdapter, NgenConfigAdapter
        assert NGENConfigAdapter is NgenConfigAdapter

    def test_adapter_creation(self):
        from symfluence.models.ngen.config import NgenConfigAdapter
        adapter = NgenConfigAdapter()
        assert adapter is not None

    def test_get_config_schema_returns_ngen_config(self):
        from symfluence.models.ngen.config import NgenConfigAdapter
        adapter = NgenConfigAdapter()
        schema = adapter.get_config_schema()
        assert schema is not None
        assert schema.__name__ == "NGENConfig"

    def test_validate_passes_with_module_params(self):
        from symfluence.models.ngen.config import NgenConfigAdapter
        adapter = NgenConfigAdapter()
        config = {
            "NGEN_MODULES_TO_CALIBRATE": "CFE",
            "NGEN_CFE_PARAMS_TO_CALIBRATE": "bb,satdk",
        }
        adapter.validate(config)

    def test_validate_raises_on_missing_module_params(self):
        from symfluence.models.ngen.config import NgenConfigAdapter
        adapter = NgenConfigAdapter()
        config = {
            "NGEN_MODULES_TO_CALIBRATE": "CFE",
            # Missing NGEN_CFE_PARAMS_TO_CALIBRATE
        }
        with pytest.raises(ConfigValidationError, match="CFE"):
            adapter.validate(config)

    def test_validate_accepts_nested_alias_module_name(self):
        from symfluence.models.ngen.config import NgenConfigAdapter

        adapter = NgenConfigAdapter()
        config = {
            "model": {
                "ngen": {
                    "modules_selected": "SLOTH,PET,SAC-SMA",
                    "modules_to_calibrate": "SAC-SMA",
                    "sacsma_params_to_calibrate": "UZTWM,UZFWM",
                }
            }
        }

        adapter.validate(config)

    def test_validate_raises_when_calibration_module_not_selected_with_aliases(self):
        from symfluence.models.ngen.config import NgenConfigAdapter

        adapter = NgenConfigAdapter()
        config = {
            "model": {
                "ngen": {
                    "modules_selected": "SLOTH,PET,CFE",
                    "modules_to_calibrate": "NOAH-OWP",
                    "noah_params_to_calibrate": "REFKDT",
                }
            }
        }

        with pytest.raises(ConfigValidationError, match="NGEN_MODULES_TO_CALIBRATE"):
            adapter.validate(config)

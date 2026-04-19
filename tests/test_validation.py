from __future__ import annotations

import pytest

from irregpca import IrregPCAConfig
from irregpca.utils.validation import validate_hyperparams


class TestValidateHyperparams:
    def test_valid(self):
        validate_hyperparams(
            n_components=2,
            epochs=100,
            lr=1e-3,
            patience=50,
            valid_split=0.2,
            batch_size=None,
            num_workers=0,
            quadrature_points=256,
            validation_frequency=1,
            training_mode="full_batch",
            integration_mode="grid",
        )

    def test_n_components_zero(self):
        # k=0 is valid (mean-only model)
        validate_hyperparams(
            n_components=0,
            epochs=100,
            lr=1e-3,
            patience=50,
            valid_split=0.2,
            batch_size=None,
            num_workers=0,
            quadrature_points=256,
            validation_frequency=1,
            training_mode="full_batch",
            integration_mode="grid",
        )

    def test_n_components_negative(self):
        with pytest.raises(ValueError, match="n_components"):
            validate_hyperparams(
                n_components=-1,
                epochs=100,
                lr=1e-3,
                patience=50,
                valid_split=0.2,
                batch_size=None,
                num_workers=0,
                quadrature_points=256,
                validation_frequency=1,
                training_mode="full_batch",
                integration_mode="grid",
            )

    def test_epochs_zero(self):
        with pytest.raises(ValueError, match="epochs"):
            validate_hyperparams(
                n_components=1,
                epochs=0,
                lr=1e-3,
                patience=50,
                valid_split=0.2,
                batch_size=None,
                num_workers=0,
                quadrature_points=256,
                validation_frequency=1,
                training_mode="full_batch",
                integration_mode="grid",
            )

    def test_lr_negative(self):
        with pytest.raises(ValueError, match="lr"):
            validate_hyperparams(
                n_components=1,
                epochs=100,
                lr=-0.01,
                patience=50,
                valid_split=0.2,
                batch_size=None,
                num_workers=0,
                quadrature_points=256,
                validation_frequency=1,
                training_mode="full_batch",
                integration_mode="grid",
            )

    def test_patience_negative(self):
        with pytest.raises(ValueError, match="patience"):
            validate_hyperparams(
                n_components=1,
                epochs=100,
                lr=1e-3,
                patience=-1,
                valid_split=0.2,
                batch_size=None,
                num_workers=0,
                quadrature_points=256,
                validation_frequency=1,
                training_mode="full_batch",
                integration_mode="grid",
            )

    def test_valid_split_zero(self):
        with pytest.raises(ValueError, match="valid_split"):
            validate_hyperparams(
                n_components=1,
                epochs=100,
                lr=1e-3,
                patience=50,
                valid_split=0.0,
                batch_size=None,
                num_workers=0,
                quadrature_points=256,
                validation_frequency=1,
                training_mode="full_batch",
                integration_mode="grid",
            )

    def test_valid_split_one(self):
        with pytest.raises(ValueError, match="valid_split"):
            validate_hyperparams(
                n_components=1,
                epochs=100,
                lr=1e-3,
                patience=50,
                valid_split=1.0,
                batch_size=None,
                num_workers=0,
                quadrature_points=256,
                validation_frequency=1,
                training_mode="full_batch",
                integration_mode="grid",
            )

    def test_invalid_training_mode(self):
        with pytest.raises(ValueError, match="training_mode"):
            validate_hyperparams(
                n_components=1,
                epochs=100,
                lr=1e-3,
                patience=50,
                valid_split=0.2,
                batch_size=None,
                num_workers=0,
                quadrature_points=256,
                validation_frequency=1,
                training_mode="batch",
                integration_mode="grid",
            )

    def test_invalid_integration_mode(self):
        with pytest.raises(ValueError, match="integration_mode"):
            validate_hyperparams(
                n_components=1,
                epochs=100,
                lr=1e-3,
                patience=50,
                valid_split=0.2,
                batch_size=None,
                num_workers=0,
                quadrature_points=256,
                validation_frequency=1,
                training_mode="full_batch",
                integration_mode="trapezoid",
            )


class TestIrregPCAConfig:
    def test_valid_config(self):
        cfg = IrregPCAConfig(n_components=3, epochs=100)
        assert cfg.n_components == 3
        assert cfg.epochs == 100

    def test_config_invalid_n_components(self):
        with pytest.raises(ValueError, match="n_components"):
            IrregPCAConfig(n_components=-1)

    def test_config_n_components_zero(self):
        cfg = IrregPCAConfig(n_components=0, epochs=10)
        assert cfg.n_components == 0

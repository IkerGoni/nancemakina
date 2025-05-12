import pytest
import os
import yaml
import time
import asyncio
from unittest.mock import patch, mock_open

from src.config_loader import ConfigManager, DEFAULT_CONFIG_PATH, EXAMPLE_CONFIG_PATH

@pytest.fixture
def temp_config_files(tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    default_config_file = config_dir / "config.yaml"
    example_config_file = config_dir / "config.yaml.example"

    example_content = {
        "api": {"key": "example_key", "secret": "example_secret"},
        "logging": {"level": "INFO"}
    }
    with open(example_config_file, "w") as f:
        yaml.dump(example_content, f)
    
    # Initially, no default config.yaml, so it should be copied from example
    return str(default_config_file), str(example_config_file)

@pytest.fixture
def cleanup_singleton():
    # Reset the singleton instance before and after each test
    if hasattr(ConfigManager, "_instance"):
        ConfigManager._instance = None
    if hasattr(ConfigManager, "_initialized"):
        # This is a bit of a hack; ideally, the singleton is designed to be reset or re-initialized for tests.
        # For this specific implementation, clearing _instance is the main thing.
        # If __init__ has instance checks, we might need to delattr(ConfigManager, "_initialized") too.
        pass 
    yield
    if hasattr(ConfigManager, "_instance"):
        ConfigManager._instance = None

@pytest.mark.asyncio
async def test_config_manager_singleton(cleanup_singleton):
    cm1 = ConfigManager(config_file_path="dummy_path1.yaml", auto_reload=False)
    cm2 = ConfigManager(config_file_path="dummy_path2.yaml", auto_reload=False)
    assert cm1 is cm2
    # Even if params are different after first init, it should return the same instance
    # and not re-initialize with new params. This is typical singleton behavior.
    # The current ConfigManager init logic ensures it only initializes fully once.
    assert cm1.config_file_path == "dummy_path1.yaml" # Should retain path from first call

@pytest.mark.asyncio
async def test_config_load_from_example_if_default_missing(temp_config_files, cleanup_singleton):
    default_cfg_path, example_cfg_path = temp_config_files
    
    # Ensure default_cfg_path does not exist initially for this part of the test
    if os.path.exists(default_cfg_path):
        os.remove(default_cfg_path)

    # Mock the global paths to use temp_config_files
    with patch("src.config_loader.DEFAULT_CONFIG_PATH", default_cfg_path), \
         patch("src.config_loader.EXAMPLE_CONFIG_PATH", example_cfg_path):
        
        cm = ConfigManager(auto_reload=False) # Should trigger copy from example
        assert os.path.exists(default_cfg_path) # Default should have been created
        config_data = cm.get_config()
        assert config_data["api"]["key"] == "example_key"
        assert config_data["logging"]["level"] == "INFO"

@pytest.mark.asyncio
async def test_config_load_existing_default(temp_config_files, cleanup_singleton):
    default_cfg_path, example_cfg_path = temp_config_files

    # Create a specific default config.yaml
    default_content = {
        "api": {"key": "default_key", "secret": "default_secret"},
        "logging": {"level": "DEBUG"}
    }
    with open(default_cfg_path, "w") as f:
        yaml.dump(default_content, f)

    with patch("src.config_loader.DEFAULT_CONFIG_PATH", default_cfg_path), \
         patch("src.config_loader.EXAMPLE_CONFIG_PATH", example_cfg_path):
        cm = ConfigManager(auto_reload=False)
        config_data = cm.get_config()
        assert config_data["api"]["key"] == "default_key"
        assert config_data["logging"]["level"] == "DEBUG"

@pytest.mark.asyncio
async def test_get_specific_config(temp_config_files, cleanup_singleton):
    default_cfg_path, example_cfg_path = temp_config_files
    # Let it load from example
    if os.path.exists(default_cfg_path):
        os.remove(default_cfg_path)

    with patch("src.config_loader.DEFAULT_CONFIG_PATH", default_cfg_path), \
         patch("src.config_loader.EXAMPLE_CONFIG_PATH", example_cfg_path):
        cm = ConfigManager(auto_reload=False)
        assert cm.get_specific_config("api.key") == "example_key"
        assert cm.get_specific_config("logging.level") == "INFO"
        assert cm.get_specific_config("non.existent.path", "default_val") == "default_val"
        assert cm.get_specific_config("api.non_existent_key") is None

@pytest.mark.asyncio
async def test_config_hot_reload(temp_config_files, cleanup_singleton):
    default_cfg_path, example_cfg_path = temp_config_files
    # Start with example content
    if os.path.exists(default_cfg_path):
        os.remove(default_cfg_path)

    callback_triggered = False
    new_conf_in_callback = None

    def my_callback(new_config):
        nonlocal callback_triggered, new_conf_in_callback
        callback_triggered = True
        new_conf_in_callback = new_config

    with patch("src.config_loader.DEFAULT_CONFIG_PATH", default_cfg_path), \
         patch("src.config_loader.EXAMPLE_CONFIG_PATH", example_cfg_path):
        
        cm = ConfigManager(auto_reload=True) # Enable auto_reload
        cm.register_callback(my_callback)
        
        initial_config = cm.get_config()
        assert initial_config["logging"]["level"] == "INFO"

        # Modify the config file
        modified_content = {
            "api": {"key": "modified_key", "secret": "modified_secret"},
            "logging": {"level": "DEBUG_MODIFIED"}
        }
        # Wait a moment to ensure the watcher is established before writing
        await asyncio.sleep(0.2) 
        with open(default_cfg_path, "w") as f:
            yaml.dump(modified_content, f)
        
        # Give watchdog time to detect and process the change
        await asyncio.sleep(1.0) # Increased sleep for reliability in CI/slower systems

        assert callback_triggered is True
        assert new_conf_in_callback is not None
        assert new_conf_in_callback["logging"]["level"] == "DEBUG_MODIFIED"
        
        current_config_from_cm = cm.get_config()
        assert current_config_from_cm["logging"]["level"] == "DEBUG_MODIFIED"
        assert cm.get_specific_config("api.key") == "modified_key"

        cm.stop_watcher() # Clean up watcher thread

@pytest.mark.asyncio
async def test_config_load_failure_empty_file(temp_config_files, cleanup_singleton):
    default_cfg_path, example_cfg_path = temp_config_files
    # Create an empty config.yaml
    with open(default_cfg_path, "w") as f:
        f.write("")

    with patch("src.config_loader.DEFAULT_CONFIG_PATH", default_cfg_path), \
         patch("src.config_loader.EXAMPLE_CONFIG_PATH", example_cfg_path):
        cm = ConfigManager(auto_reload=False)
        # Should log a warning, and config_data should be empty or fallback
        # The current implementation falls back to empty dict if load fails post-initialization
        assert cm.get_config() == {} 

@pytest.mark.asyncio
async def test_config_load_failure_invalid_yaml(temp_config_files, cleanup_singleton):
    default_cfg_path, example_cfg_path = temp_config_files
    # Create an invalid YAML config.yaml
    with open(default_cfg_path, "w") as f:
        f.write("api: key: -: invalid_yaml_structure")

    # Pre-populate with valid example data first so there is an "old" config
    example_data = {"api": {"key": "example_key"}}
    with open(example_cfg_path, "w") as f:
        yaml.dump(example_data, f)
    if os.path.exists(default_cfg_path):
        os.remove(default_cfg_path)

    with patch("src.config_loader.DEFAULT_CONFIG_PATH", default_cfg_path), \
         patch("src.config_loader.EXAMPLE_CONFIG_PATH", example_cfg_path):
        
        cm_initial = ConfigManager(auto_reload=False) # Loads from example
        initial_key = cm_initial.get_specific_config("api.key")
        assert initial_key == "example_key"

        # Now, simulate the invalid file being loaded (e.g., by a hot reload attempt or direct load)
        with open(default_cfg_path, "w") as f:
            f.write("api: key: -: invalid_yaml_structure")
        
        cm_initial.load_config() # Manually trigger a load of the bad file

        # Config should remain the old valid one due to parsing error
        assert cm_initial.get_specific_config("api.key") == initial_key


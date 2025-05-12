import yaml
import os
import logging
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from threading import Lock

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "config.yaml")
EXAMPLE_CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "config.yaml.example")

class ConfigChangeHandler(FileSystemEventHandler):
    def __init__(self, config_manager):
        self.config_manager = config_manager

    def on_modified(self, event):
        if not event.is_directory and event.src_path == self.config_manager.config_file_path:
            logger.info(f"Configuration file {event.src_path} changed. Reloading...")
            self.config_manager.load_config()

class ConfigManager:
    _instance = None
    _lock = Lock()

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super(ConfigManager, cls).__new__(cls)
        return cls._instance

    def __init__(self, config_file_path=None, auto_reload=True):
        # Ensure __init__ is only run once for the singleton instance
        if not hasattr(self, '_initialized'):
            with self._lock:
                if not hasattr(self, '_initialized'): 
                    self.config_file_path = config_file_path or DEFAULT_CONFIG_PATH
                    self.config_data = {}
                    self._callbacks = []
                    self.observer = None
                    self.auto_reload = auto_reload
                    self._ensure_config_file_exists()
                    self.load_config()
                    if self.auto_reload:
                        self._start_watcher()
                    self._initialized = True

    def _ensure_config_file_exists(self):
        if not os.path.exists(self.config_file_path):
            logger.warning(
                f"Config file not found at {self.config_file_path}. "
                f"Attempting to copy from {EXAMPLE_CONFIG_PATH}."
            )
            try:
                config_dir = os.path.dirname(self.config_file_path)
                if not os.path.exists(config_dir):
                    os.makedirs(config_dir)
                
                with open(EXAMPLE_CONFIG_PATH, 'r') as src, open(self.config_file_path, 'w') as dst:
                    dst.write(src.read())
                logger.info(f"Successfully copied example config to {self.config_file_path}. Please review and update it.")
            except Exception as e:
                logger.error(f"Could not copy example config: {e}. Please create {self.config_file_path} manually.")
                # Potentially raise an error or exit if config is critical for startup

    def load_config(self):
        try:
            with open(self.config_file_path, 'r') as f:
                new_config_data = yaml.safe_load(f)
            if new_config_data:
                self.config_data = new_config_data
                logger.info(f"Configuration loaded successfully from {self.config_file_path}")
                self._notify_callbacks()
            else:
                logger.warning(f"Configuration file {self.config_file_path} is empty or invalid.")
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {self.config_file_path}")
            # Fallback to empty or default config if appropriate, or raise error
            self.config_data = {}
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML configuration file {self.config_file_path}: {e}")
            # Keep old config or fallback
        except Exception as e:
            logger.error(f"An unexpected error occurred while loading configuration: {e}")

    def get_config(self):
        with self._lock: # Ensure thread-safe access to config_data
            return self.config_data.copy() # Return a copy to prevent modification

    def get_specific_config(self, key_path, default=None):
        """ Fetches a specific config value using a dot-separated key path. E.g., 'api.binance_api_key' """
        try:
            value = self.config_data
            for key in key_path.split('.'):
                if isinstance(value, dict):
                    value = value[key]
                else:
                    # If at any point value is not a dict and we still have keys, path is invalid
                    logger.warning(f"Invalid key path '{key_path}' at segment '{key}'.")
                    return default
            return value
        except KeyError:
            logger.debug(f"Config key '{key_path}' not found. Returning default: {default}")
            return default
        except Exception as e:
            logger.error(f"Error getting specific config '{key_path}': {e}")
            return default

    def register_callback(self, callback):
        if callable(callback):
            self._callbacks.append(callback)
        else:
            logger.warning("Attempted to register a non-callable callback.")

    def unregister_callback(self, callback):
        """Remove a previously registered callback."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)
            logger.debug(f"Callback {callback.__name__ if hasattr(callback, '__name__') else callback} unregistered.")
        else:
            logger.warning(f"Attempted to unregister a callback that was not registered.")

    def _notify_callbacks(self):
        for callback in self._callbacks:
            try:
                callback(self.get_config())
            except Exception as e:
                logger.error(f"Error executing config update callback {callback.__name__}: {e}")

    def _start_watcher(self):
        if self.observer:
            self.observer.stop()
            self.observer.join() # Wait for the thread to finish

        event_handler = ConfigChangeHandler(self)
        self.observer = Observer()
        # Observe the directory containing the config file, as some editors modify files by creating a new one and renaming.
        config_dir = os.path.dirname(self.config_file_path) or '.'
        self.observer.schedule(event_handler, config_dir, recursive=False)
        
        try:
            self.observer.start()
            logger.info(f"Started watching configuration file {self.config_file_path} for changes.")
        except Exception as e:
            logger.error(f"Error starting configuration file watcher: {e}")
            self.observer = None # Ensure observer is None if it failed to start

    def stop_watcher(self):
        if self.observer and self.observer.is_alive():
            self.observer.stop()
            self.observer.join()
            logger.info("Stopped configuration file watcher.")

# Example usage (typically in main.py or similar entry point):
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create a dummy config.yaml for testing
    dummy_config_content = """
    api:
      binance_api_key: "TEST_KEY"
      binance_api_secret: "TEST_SECRET"
    logging:
      level: "DEBUG"
    """
    if not os.path.exists(os.path.dirname(DEFAULT_CONFIG_PATH)):
        os.makedirs(os.path.dirname(DEFAULT_CONFIG_PATH))
    with open(DEFAULT_CONFIG_PATH, "w") as f:
        f.write(dummy_config_content)

    config_manager = ConfigManager()
    print("Initial config:", config_manager.get_config())
    print("API Key:", config_manager.get_specific_config("api.binance_api_key"))
    print("Logging Level:", config_manager.get_specific_config("logging.level"))

    def my_config_update_handler(new_config):
        print("Callback: Config updated! New logging level:", new_config.get("logging", {}).get("level"))

    config_manager.register_callback(my_config_update_handler)

    print(f"\nSimulating config file change. Please modify {DEFAULT_CONFIG_PATH} and save.")
    print("(e.g., change logging.level to INFO)")
    print("Watching for 30 seconds... Press Ctrl+C to stop early.")
    
    try:
        time.sleep(30) # Keep alive to observe changes
    except KeyboardInterrupt:
        print("\nExiting example.")
    finally:
        config_manager.stop_watcher()
        # Clean up dummy config
        # os.remove(DEFAULT_CONFIG_PATH)


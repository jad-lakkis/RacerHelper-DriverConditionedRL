"""
This file contains user-level configuration for the Wine/Linux environment.
It is expected that the user fills this file once when setting up the project, and does not need to modify it after.
"""

from pathlib import Path
import os
import platform

is_linux = platform.system() == "Linux"

username = "default"  # TMNF profile name

# Wine prefix Documents path
_wine_docs = Path("/home/wineuser/.wine/drive_c/users/wineuser/Documents")

# Path where Python_Link.as is placed so that it can be loaded in TMInterface
target_python_link_path = _wine_docs / "TMInterface" / "Plugins" / "Python_Link.as"

# TrackMania base path
trackmania_base_path = _wine_docs / "TmForever"

# Communication port for the first TMInterface instance that will be launched.
# If using multiple instances, the ports used will be base_tmi_port + 1, +2, +3, etc...
base_tmi_port = 8478

# Linux launch script — resolved relative to this file's location
linux_launch_game_path = Path(__file__).parent.parent / "scripts" / "launch_game.sh"

# Windows paths, unused on Linux
windows_TMLoader_path = Path(os.path.expanduser("~")) / "AppData" / "Local" / "TMLoader" / "TMLoader.exe"
windows_TMLoader_profile_name = "default"

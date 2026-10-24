import site
import sys
import os

# Ensure user site-packages are on sys.path for pytest execution
user_site = site.getusersitepackages()
if os.path.isdir(user_site) and user_site not in sys.path:
    sys.path.append(user_site)

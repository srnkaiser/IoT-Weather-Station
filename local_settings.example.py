"""
Template for local_settings.py - copy this file, then fill in your keys.

    cp local_settings.example.py local_settings.py

local_settings.py is git-ignored on purpose. A ThingSpeak write key in a
public repository lets anyone push data into the channel.

Reading needs no key at all as long as the measurement channel is public,
so training, evaluation and read-only live inference all work without this
file. Only writing predictions back does not.
"""

# Write API key of the prediction channel (3451792).
# Found under Channel Settings -> API Keys.
THINGSPEAK_WRITE_KEY = ""

# Read key of the measurement channel (3448191).
# Leave empty while that channel is public.
THINGSPEAK_READ_KEY = ""

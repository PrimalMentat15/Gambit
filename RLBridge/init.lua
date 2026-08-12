--- RLBridge initialization module
--- Handles initial game setup and configuration for the RL system
--- Fast mode is applied when RL training activates, not at load, so launching
--- Balatro on its own stays a normal game

local INIT = {}

-- Populated when fast mode is enabled so the player's own settings can be put
-- back. G.SETTINGS is persisted to the profile, so leaving GAMESPEED at 100
-- would follow the player into ordinary sessions.
local saved_settings = nil

-- Read by the animatedsprite patch in lovely/init.toml every time a sprite
-- animation is queued, so toggling it at runtime takes effect immediately
DISABLE_ANIMATIONS = false

--- Enable fast mode for training
--- Skips sprite animations and runs the game clock at 100x
--- @return nil
function INIT.enable_fast_mode()
    if saved_settings or not G.SETTINGS then
        return
    end

    saved_settings = {
        gamespeed = G.SETTINGS.GAMESPEED,
        reduced_motion = G.SETTINGS.reduced_motion,
    }

    DISABLE_ANIMATIONS = true
    G.SETTINGS.GAMESPEED = 100
    G.SETTINGS.reduced_motion = true

    print("[AI] Fast mode enabled (gamespeed 100, animations off)")
end

--- Restore the player's original speed settings
--- @return nil
function INIT.disable_fast_mode()
    if not saved_settings then
        return
    end

    DISABLE_ANIMATIONS = false
    G.SETTINGS.GAMESPEED = saved_settings.gamespeed
    G.SETTINGS.reduced_motion = saved_settings.reduced_motion
    saved_settings = nil

    print("[AI] Fast mode disabled (original settings restored)")
end

--- Whether fast mode is currently applied
--- @return boolean
function INIT.is_fast_mode()
    return saved_settings ~= nil
end

--- Should be called after G:start_up() to safely initialize and start a run
--- Initializes the AI system and starts a new Balatro run with default settings
--- @return nil
function INIT.start_run()
    print("Starting balatro run")

    -- Initialize AI system
    local ai = require("ai")
    ai.init()
end

return INIT

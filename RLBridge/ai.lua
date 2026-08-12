--- RLBridge AI controller module
--- Main AI loop that handles state monitoring, action execution, and file-based communication
--- Runs every frame and coordinates between game state observation and AI decisions

local AI = {}

local output = require("output")
local action = require("actions")
local communication = require("communication")
local utils = require("utils")

-- State management
local last_combined_hash = nil
local pending_action = nil
local rl_training_active = false
local last_key_pressed = nil
local retry_count = 0

-- Latency instrumentation
-- Python can only observe total wait time. These counters attribute that wait
-- to a cause: frames blocked on the event queue (animations) versus frames
-- where the game was ready but the state hash had not changed yet.
local frames_since_last_action = 0
local blocked_frames = 0
local not_ready_frames = 0
local t_last_action = nil

-- Diagnostic sampling of which events are blocking. Set BALATRO_RL_DIAG=0 to
-- skip the per-frame queue walk once the cause is understood.
local DIAG = os.getenv("BALATRO_RL_DIAG") ~= "0"
local blocking_sigs = {}

-- Event queue draining.
-- EventManager:update rate-limits itself to one pass per 1/60s of REAL time
-- (game.lua drives it with real_dt, not the GAMESPEED-scaled dt) and completes
-- at most one blocking event per pass. That caps the game at ~60 blocking
-- events per second no matter how high GAMESPEED is, which is why a scored hand
-- costs about a second. Set BALATRO_RL_DRAIN=0 to fall back to stock pacing.
local DRAIN = os.getenv("BALATRO_RL_DRAIN") ~= "0"
local MAX_DRAIN_PASSES = 500
local drain_passes = 0

-- Card movement pumping. OFF by default: measured as a net regression.
-- game.lua drives Moveable:move with real_dt capped at 1/20 and never applies
-- SPEEDFACTOR, so cards travel at real-time speed however high GAMESPEED is.
-- Pumping movement inside the drain loop does let position-gated events finish,
-- but it also stops the loop from bailing early, so it runs hundreds of passes
-- that each walk every moveable. Measured 114 ms/step versus 65 ms/step with the
-- event drain alone, with one step reaching 13 s. Set BALATRO_RL_PUMP=1 to
-- re-enable for experiments.
local PUMP = os.getenv("BALATRO_RL_PUMP") == "1"
local MOVE_DT = 1 / 20  -- the engine's own ceiling in game.lua

--- Advance all moveables by one movement step without spending a real frame
--- Mirrors the per-frame setup in game.lua so interpolation matches exactly
--- @return nil
local function pump_moveables()
    if not (G.MOVEABLES and G.FRAMES and G.exp_times) then
        return
    end

    G.exp_times.xy = math.exp(-50 * MOVE_DT)
    G.exp_times.scale = math.exp(-60 * MOVE_DT)
    G.exp_times.r = math.exp(-190 * MOVE_DT)
    G.exp_times.max_vel = 70 * MOVE_DT

    -- Moveable:move refuses to run twice within the same movement frame
    G.FRAMES.MOVE = G.FRAMES.MOVE + 1

    for _, moveable in pairs(G.MOVEABLES) do
        if moveable.FRAME.MOVE < G.FRAMES.MOVE then
            moveable:move(MOVE_DT)
        end
    end
end

--- Count blocking events still pending across all queues
--- @return number Number of incomplete blocking events
local function count_blocking()
    local total = 0
    if not (G.E_MANAGER and G.E_MANAGER.queues) then
        return 0
    end

    for _, queue in pairs(G.E_MANAGER.queues) do
        for _, event in ipairs(queue) do
            if event.blocking and not event.complete then
                total = total + 1
            end
        end
    end

    return total
end

--- Advance the event queue without spending a frame per event
--- Runs the same events in the same order; only the wall-clock spacing between
--- them changes. The game-time clock is advanced alongside so that chained
--- 'after' delays expire, since G.TIMERS.TOTAL is otherwise frozen inside this
--- loop. Stops as soon as a pass makes no progress, which means the remaining
--- events are waiting on card movement that only the main loop advances.
--- @return number Number of forced passes performed
local function drain_event_queue()
    if not (G.E_MANAGER and G.E_MANAGER.update) then
        return 0
    end

    local dt = G.real_dt or (1 / 60)
    local speed = G.SPEEDFACTOR or (G.SETTINGS and G.SETTINGS.GAMESPEED) or 1
    local passes = 0
    local pending = count_blocking()

    while pending > 0 and passes < MAX_DRAIN_PASSES do
        -- Mirror the clock advance G:update performs, so timer-gated events
        -- become eligible instead of stalling until the next real frame
        G.TIMERS.TOTAL = G.TIMERS.TOTAL + dt * speed

        -- Let cards reach their targets, otherwise position-gated events
        -- never complete and the loop bails after a single pass
        if PUMP then
            pump_moveables()
        end

        G.E_MANAGER:update(dt, true)
        passes = passes + 1

        local remaining = count_blocking()
        if remaining >= pending then
            break
        end
        pending = remaining
    end

    return passes
end

--- Reset the per-request timing counters
--- @return nil
local function reset_timing()
    frames_since_last_action = 0
    blocked_frames = 0
    not_ready_frames = 0
    drain_passes = 0
    blocking_sigs = {}
    t_last_action = love and love.timer and love.timer.getTime() or 0
end

--- Record which events are blocking on this frame
--- Counts frames per distinct event signature so the report can rank them
--- @return nil
local function sample_blocking()
    for _, event in ipairs(utils.describe_blocking_events(6)) do
        local key = event.queue .. "|" .. event.trigger .. "|timer=" ..
            event.timer .. "|delay=" .. tostring(event.delay)
        blocking_sigs[key] = (blocking_sigs[key] or 0) + 1
    end
end

--- Convert the signature counts into a JSON-friendly array
--- @return table Array of {sig, frames} sorted by frame count
local function collect_blocking()
    local list = {}
    for sig, frames in pairs(blocking_sigs) do
        table.insert(list, { sig = sig, frames = frames })
    end
    table.sort(list, function(a, b) return a.frames > b.frames end)
    return list
end

--- Snapshot the current timing counters for sending to the AI
--- @return table Timing data describing what the game did since the last action
local function collect_timing()
    local now = love and love.timer and love.timer.getTime() or 0
    return {
        t_frame = now,
        elapsed = t_last_action and (now - t_last_action) or 0,
        frames_since_last_action = frames_since_last_action,
        blocked_frames = blocked_frames,
        not_ready_frames = not_ready_frames,
        gamespeed = G.SETTINGS and G.SETTINGS.GAMESPEED or 0,
        drain_passes = drain_passes,
        blocking = collect_blocking(),
    }
end

--- Initialize AI system
--- Sets up communication and prepares the AI for operation
--- @return nil
function AI.init()
    communication.init()

    -- Hook into Love2D keyboard events
    if love and love.keypressed then
        local original_keypressed = love.keypressed
        love.keypressed = function(key)
            -- Store the key press for our AI
            last_key_pressed = key

            -- Call original function
            if original_keypressed then
                original_keypressed(key)
            end
        end
    end
end

--- Main AI update loop (called every frame)
--- Monitors game state changes, handles communication, and executes AI actions
--- @return nil
function AI.update()
    -- Check for key press to start/stop RL training
    if last_key_pressed then
        if last_key_pressed == "r" then
            if not rl_training_active then
                rl_training_active = true
                reset_timing()
                utils.log_ai("\n\nRL Training STARTED (R pressed)")
            end
        end
        -- Clear the key press
        last_key_pressed = nil
    end

    -- Don't process AI requests unless RL training is active
    if not rl_training_active then
        return
    end

    -- Attribute every frame spent waiting for the game to settle. Measurement
    -- only: these counters never gate whether an action is sent.
    frames_since_last_action = frames_since_last_action + 1
    if utils.has_blocking_events() then
        blocked_frames = blocked_frames + 1
        if DIAG then
            sample_blocking()
        end
        -- Collapse the animation backlog into this frame instead of waiting
        -- one frame per event
        if DRAIN then
            drain_passes = drain_passes + drain_event_queue()
        end
    end
    if not utils.is_game_ready_for_action() then
        not_ready_frames = not_ready_frames + 1
    end

    -- Get current game state
    local current_state = output.get_game_state()
    local available_actions = action.get_available_actions()

    -- Don't continue if state = -1
    if current_state.state == -1 then
        return
    end

    -- Don't continue if there are no actions for the AI to do
    if next(available_actions) == nil then
        return
    end

    -- Create combined hash to detect meaningful changes
    local combined_hash = AI.hash_combined_state(current_state, available_actions)

    if combined_hash ~= last_combined_hash then
        -- Game state or available actions have changed
        utils.log_ai("State/Actions changed: State: " ..
            current_state.state .. " (" .. utils.get_state_name(current_state.state) .. ") | " ..
            "Actions: " .. table.concat(utils.get_action_names(available_actions), ", "))

        action.reset_state()

        -- Auto-skip trivial actions (don't send to AI)
        if AI.should_auto_skip(current_state, available_actions) then
            AI.execute_auto_skip_action(current_state, available_actions)
            return
        end

        -- Add retry_count to current state
        current_state.retry_count = retry_count
        
        -- Request action from AI (only for core gameplay)
        local ai_response = communication.request_action(current_state, available_actions, collect_timing())

        if ai_response then
            pending_action = ai_response
            last_combined_hash = combined_hash
        end
    end

    -- Execute pending action
    if pending_action then
        -- Validate action is still available in current state
        local current_actions = action.get_available_actions()
        local action_still_valid = false
        for _, valid_action in ipairs(current_actions) do
            if valid_action == pending_action.action then
                action_still_valid = true
                break
            end
        end
        
        if action_still_valid then
            local result = action.execute_action(pending_action.action, pending_action.params)
            if result.success then
                utils.log_ai("Action executed successfully: " .. pending_action.action)
                retry_count = 0  -- Reset retry count on success
                pending_action = nil
                reset_timing()
                utils.log_ai("\n\n\n")
            else
                utils.log_ai("Action failed: " .. (result.error or "Unknown error"))
                retry_count = retry_count + 1
                utils.log_ai("Retry count: " .. retry_count)
                -- Keep pending_action to retry on next frame
                -- Force state recheck to send updated state with retry_count
                last_combined_hash = nil
            end
        else
            utils.log_ai("Action no longer valid (state changed), discarding: " .. pending_action.action)
            retry_count = 0  -- Reset on state change
            pending_action = nil
            utils.log_ai("\n\n\n")
        end
    end
end

--- Create combined hash of game state and actions for change detection
--- Only sends AI requests when both state and actions are ready/changed
--- @param game_state table Current game state data
--- @param available_actions table Available actions list
--- @return string Combined hash representing state + actions
function AI.hash_combined_state(game_state, available_actions)
    -- State components
    local state_parts = {
        game_state.state or 0,
        game_state.chips or 0,
        game_state.blind_chips or 0,
        (game_state.round and game_state.round.hands_left) or 0,
        (game_state.round and game_state.round.discards_left) or 0,
        (game_state.hand and game_state.hand.size) or 0,
        (game_state.hand and game_state.hand.highlighted_count) or 0,
        game_state.game_over or 0
    }

    -- Action components
    local action_ids = {}
    for _, id in ipairs(available_actions) do
        table.insert(action_ids, tostring(id))
    end
    table.sort(action_ids)

    -- Combine everything
    local combined = table.concat(state_parts, "_") .. "|" .. table.concat(action_ids, ",")
    return combined
end

--- Check if current state should be auto-skipped (not sent to AI)
--- @param current_state table Current game state  
--- @param available_actions table Available actions list
--- @return boolean True if should auto-skip, false if send to AI
function AI.should_auto_skip(current_state, available_actions)
    -- Auto-skip START_RUN in menu (action ID = 4)
    if current_state.state == G.STATES.MENU and #available_actions == 1 and available_actions[1] == 4 then
        return true
    end
    
    -- Auto-skip SELECT_BLIND in blind selection (action ID = 5)
    if current_state.state == G.STATES.BLIND_SELECT and #available_actions == 1 and available_actions[1] == 5 then
        return true
    end
    
    -- Don't auto-skip anything else - core actions (1,2,3) go to AI
    
    return false
end

--- Execute auto-skip action without AI involvement
--- @param current_state table Current game state
--- @param available_actions table Available actions list  
function AI.execute_auto_skip_action(current_state, available_actions)
    local action_id = available_actions[1]
    utils.log_ai("Auto-executing action: " .. action.get_action_name(action_id))
    
    local result = action.execute_action(action_id, {})
    if result.success then
        utils.log_ai("Auto-execution successful: " .. action.get_action_name(action_id))
        reset_timing()
    else
        utils.log_ai("Auto-execution failed: " .. (result.error or "Unknown error"))
    end
    
    -- Force state recheck after auto-execution
    last_combined_hash = nil
end

return AI

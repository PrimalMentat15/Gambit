--- RLBridge communication module
--- Handles line-delimited JSON communication with the external AI system over a
--- localhost TCP socket. The Python trainer is the server; the game is the client.

local COMM = {}
local utils = require("utils")
local json = require("dkjson")

-- LuaSocket ships inside LÖVE; fall back to the raw core module if the
-- pure-Lua wrapper is unavailable
local ok, socket = pcall(require, "socket")
if not ok then
    ok, socket = pcall(require, "socket.core")
end
if not ok then
    socket = nil
end

-- Socket communication settings with a persistent connection
local comm_enabled = false
local host = os.getenv("BALATRO_RL_HOST") or "127.0.0.1"
local port = tonumber(os.getenv("BALATRO_RL_PORT") or "12345")
local conn = nil
local last_connect_attempt = 0
local RECONNECT_COOLDOWN = 1 -- seconds between connect attempts

--- Initialize socket communication
--- Sets up a persistent connection with the external AI system
--- @return nil
function COMM.init()
    utils.log_comm("Initializing socket communication...")
    comm_enabled = true -- Enable communication, connection opens on first use
end

--- Lazy initialization of the socket connection when first needed
--- @return boolean True if the connection is ready, false otherwise
function COMM.ensure_connected()
    if conn then
        return true -- Already connected
    end

    if not socket then
        utils.log_comm("ERROR: LuaSocket unavailable, cannot connect")
        return false
    end

    -- AI.update runs every frame; throttle retries so a missing trainer
    -- does not stall the game with a connect attempt per frame
    local now = os.time()
    if now - last_connect_attempt < RECONNECT_COOLDOWN then
        return false
    end
    last_connect_attempt = now

    local sock, err = socket.tcp()
    if not sock then
        utils.log_comm("ERROR: Cannot create socket: " .. tostring(err))
        return false
    end

    -- Bounded connect so a missing trainer does not hang the game forever
    sock:settimeout(2)
    local connected, connect_err = sock:connect(host, port)
    if not connected then
        utils.log_comm("ERROR: Cannot connect to AI at " .. host .. ":" .. port ..
            " (" .. tostring(connect_err) .. ")")
        sock:close()
        return false
    end

    -- Block indefinitely on reads; the AI decides when it is done thinking
    sock:settimeout(nil)
    pcall(function() sock:setoption("tcp-nodelay", true) end)
    conn = sock

    utils.log_comm("Connected to AI at " .. host .. ":" .. port)
    return true
end

--- Send game turn request to AI and get action via the persistent socket
--- @param game_state table Current game state data
--- @param available_actions table Available actions list
--- @param timing table|nil Game-side latency counters since the last action
--- @return table|nil Action response from AI, nil if error
function COMM.request_action(game_state, available_actions, timing)
    if not comm_enabled then
        utils.log_comm("ERROR: Communication not enabled")
        return nil
    end

    -- Lazy initialization - connect when first needed
    if not COMM.ensure_connected() then
        utils.log_comm("ERROR: Failed to establish connection")
        return nil
    end

    local request = {
        game_state = game_state,
        available_actions = available_actions or {},
        timing = timing or {},
    }

    utils.log_comm(utils.get_timestamp() .. " Sending action request for state: " ..
        tostring(game_state.state) .. " (" .. utils.get_state_name(game_state.state) .. ")")

    -- Encode request as JSON
    local json_data = json.encode(request)
    if not json_data then
        utils.log_comm("ERROR: Failed to encode request JSON")
        return nil
    end

    -- Write request to the socket
    local sent, send_err = conn:send(json_data .. "\n")
    if not sent then
        utils.log_comm("ERROR: Failed to send request: " .. tostring(send_err))
        COMM.drop_connection()
        return nil
    end

    -- Read response from the socket
    local response_json, recv_err = conn:receive("*l")

    if not response_json or response_json == "" then
        utils.log_comm("ERROR: No response received from AI: " .. tostring(recv_err))
        COMM.drop_connection()
        return nil
    end

    local response_data = json.decode(response_json)
    if not response_data then
        utils.log_comm("ERROR: Failed to decode response JSON")
        return nil
    end

    utils.log_comm(utils.get_timestamp() .. " AI action: " .. tostring(response_data.action))
    return response_data
end

--- Drop a dead connection so the next request reconnects
--- @return nil
function COMM.drop_connection()
    if conn then
        conn:close()
        conn = nil
    end
end

--- Check if socket communication is enabled
--- Returns the current communication status
--- @return boolean True if enabled, false otherwise
function COMM.is_connected()
    return comm_enabled
end

--- Close communication
--- Terminates the persistent connection with the AI system
--- @return nil
function COMM.close()
    comm_enabled = false
    COMM.drop_connection()
    utils.log_comm("Socket communication closed")
end

return COMM

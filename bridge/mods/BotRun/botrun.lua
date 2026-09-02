-- BOT RUN button + deck/stake pickers for the main menu.
--
-- The DECK and STAKE cycle selectors choose which run the bot plays. Clicking
-- BOT RUN writes "<DECK> <STAKE>" (uppercase names, e.g. "PLASMA GOLD") via
-- love.filesystem into the game's save directory (under Proton:
-- .../compatdata/2379780/pfx/drive_c/users/steamuser/AppData/Roaming/Balatro/).
-- The `balatro-bridge-autoplay` daemon (bridge/, Python) polls for that marker
-- and then starts + plays a random-seed run of THAT deck/stake through the
-- balatrobot API with the trained policy. The button itself starts nothing, so
-- a click with no daemon running is a harmless no-op.

local REQUEST_FILE = "botrun_request.txt"
local USER_QUIT_FILE = "botrun_user_quit.txt"

-- UPPERCASE names, matching the sim's export_deck_name / export_stake_name and
-- the bridge's balatro_bridge.types Literals -- written verbatim to the marker,
-- so this list is the single source the daemon parses.
local BOTRUN_DECKS = {
    "RED", "BLUE", "YELLOW", "GREEN", "BLACK", "MAGIC", "NEBULA", "GHOST",
    "ABANDONED", "CHECKERED", "ZODIAC", "PAINTED", "ANAGLYPH", "PLASMA", "ERRATIC",
}
local BOTRUN_STAKES = {
    "WHITE", "RED", "GREEN", "BLACK", "BLUE", "PURPLE", "ORANGE", "GOLD",
}
local botrun_deck = "RED"
local botrun_stake = "WHITE"

-- Tell the supervising launcher this is an intentional quit (menu QUIT button
-- or closing the window -- both fire love.quit; a crash doesn't), so it stops
-- instead of relaunching the game.
local botrun_orig_love_quit = love.quit
function love.quit()
    love.filesystem.write(USER_QUIT_FILE, tostring(os.time()))
    if botrun_orig_love_quit then
        return botrun_orig_love_quit()
    end
end

local function botrun_toast(text)
    if G.play then
        attention_text({
            text = text,
            scale = 0.7,
            hold = 2,
            major = G.play,
            backdrop_colour = G.C.SECONDARY_SET.Planet,
            align = "cm",
            offset = { x = 0, y = -3.5 },
            silent = true,
        })
    end
end

-- create_option_cycle passes the selection back as {to_key = index, to_val =
-- options[index]}; older builds pass only to_key, so resolve from either.
G.FUNCS.botrun_deck_cycle = function(args)
    if args then
        botrun_deck = args.to_val or BOTRUN_DECKS[args.to_key] or botrun_deck
    end
end
G.FUNCS.botrun_stake_cycle = function(args)
    if args then
        botrun_stake = args.to_val or BOTRUN_STAKES[args.to_key] or botrun_stake
    end
end

G.FUNCS.botrun_click = function(e)
    love.filesystem.write(REQUEST_FILE, botrun_deck .. " " .. botrun_stake)
    play_sound("coin2")
    botrun_toast("BOT RUN: " .. botrun_deck .. " / " .. botrun_stake)
end

-- Game speed under balatrobot. balatrobot pins the game to a FIXED dt of 1/60
-- per rendered frame (its settings.lua patches love.update), so wall-clock speed
-- is set by the FRAME RATE, not G.SETTINGS.GAMESPEED: at N*60 FPS the sim
-- advances N x real time. And most animation waits are coded as
-- delay = base*GAMESPEED on a clock that also scales by GAMESPEED, so they take
-- a constant NUMBER OF FRAMES -- which is why raising GAMESPEED alone (the old
-- behaviour) barely changed anything. So drive the FPS cap (the real lever) and
-- GAMESPEED together. Actual speed is still bounded by how many FPS the machine
-- can render; past that ceiling higher buttons look identical -- hardware limit,
-- not a bug. balatrobot reads state between animations, so back off if the
-- bridge starts stalling / diverging.
for _, n in ipairs({ 8, 16, 32, 64 }) do
    G.FUNCS["botrun_speed_" .. n] = function(e)
        G.FPS_CAP = 60 * n
        G.SETTINGS.GAMESPEED = n
        play_sound("button")
        botrun_toast(n .. "X SPEED")
    end
end

local function botrun_speed_button(n, colour)
    return UIBox_button({
        id = "botrun_speed_" .. n,
        button = "botrun_speed_" .. n,
        colour = colour,
        minw = 1.0,
        minh = 0.6,
        label = { n .. "X" },
        scale = 0.32,
        col = true,
    })
end

local function botrun_spacer(w)
    return { n = G.UIT.C, config = { minw = w }, nodes = {} }
end

local botrun_orig_main_menu = create_UIBox_main_menu_buttons
function create_UIBox_main_menu_buttons()
    local t = botrun_orig_main_menu()
    -- t.nodes[1] is the C holding the button bar (UI_definitions.lua:6212-6222).
    -- The bar's PLAY/(OPTIONS|QUIT)/COLLECTION buttons are C nodes, which the UI
    -- engine lays out side by side (ui.lua:187-198) -- inserting another C
    -- widens the bar past the screen. R children stack vertically instead, so
    -- the bot panel goes in as its own rows above the bar, leaving it untouched.
    local holder = t.nodes and t.nodes[1]
    if holder and holder.nodes then
        -- The panel bottom-anchors above the button bar and grows UPWARD, so it
        -- must stay short or its top rows slide behind the deck-preview card.
        -- DECK + STAKE therefore share ONE row (side by side), keeping the whole
        -- panel to three rows -- the height the button-only version cleared.
        table.insert(holder.nodes, 1, {
            n = G.UIT.R,
            config = { align = "cm", padding = 0.1, r = 0.1, emboss = 0.1, colour = G.C.L_BLACK },
            nodes = {
                { n = G.UIT.R, config = { align = "cm", padding = 0.05 }, nodes = {
                    { n = G.UIT.C, config = { align = "cm", padding = 0.06 }, nodes = {
                        create_option_cycle({
                            label = "DECK",
                            scale = 0.5,
                            w = 2.6,
                            options = BOTRUN_DECKS,
                            current_option = 1,
                            opt_callback = "botrun_deck_cycle",
                        }),
                    } },
                    { n = G.UIT.C, config = { align = "cm", padding = 0.06 }, nodes = {
                        create_option_cycle({
                            label = "STAKE",
                            scale = 0.5,
                            w = 2.6,
                            options = BOTRUN_STAKES,
                            current_option = 1,
                            opt_callback = "botrun_stake_cycle",
                        }),
                    } },
                } },
                { n = G.UIT.R, config = { align = "cm", padding = 0.06 }, nodes = {
                    UIBox_button({
                        id = "botrun_button",
                        button = "botrun_click",
                        colour = G.C.GREEN,
                        minw = 3.65,
                        minh = 0.8,
                        label = { "BOT RUN" },
                        scale = 0.45,
                        col = true,
                    }),
                } },
                { n = G.UIT.R, config = { align = "cm" }, nodes = {
                    botrun_speed_button(8, G.C.ORANGE),
                    botrun_spacer(0.1),
                    botrun_speed_button(16, G.C.RED),
                    botrun_spacer(0.1),
                    botrun_speed_button(32, G.C.PURPLE),
                    botrun_spacer(0.1),
                    botrun_speed_button(64, G.C.BLUE),
                } },
            },
        })
        table.insert(holder.nodes, 2, { n = G.UIT.R, config = { minh = 0.15 }, nodes = {} })
    end
    return t
end

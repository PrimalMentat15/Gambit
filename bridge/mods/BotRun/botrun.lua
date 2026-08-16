-- BOT RUN button for the main menu.
--
-- Clicking it writes a one-line request marker via love.filesystem into the
-- game's save directory (under Proton:
-- .../compatdata/2379780/pfx/drive_c/users/steamuser/AppData/Roaming/Balatro/).
-- The `balatro-bridge-autoplay` daemon (bridge/, Python) polls for that file
-- and then starts + plays a random-seed Red Deck / White Stake run through
-- the balatrobot API with the trained policy. The button itself starts
-- nothing, so a click with no daemon running is a harmless no-op.

local REQUEST_FILE = "botrun_request.txt"
local USER_QUIT_FILE = "botrun_user_quit.txt"

-- Tell the supervising launcher this is an intentional quit (menu QUIT
-- button or closing the window — both fire love.quit; a crash doesn't), so
-- it stops instead of relaunching the game.
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

G.FUNCS.botrun_click = function(e)
    love.filesystem.write(REQUEST_FILE, tostring(os.time()))
    play_sound("coin2")
    botrun_toast("BOT RUN QUEUED")
end

-- Game speed. The vanilla settings UI caps G.SETTINGS.GAMESPEED at 4, but
-- the engine takes any multiplier (balatrobot boots it from --gamespeed,
-- default 4 — these buttons override it live for the session).
for _, n in ipairs({ 4, 8, 16 }) do
    G.FUNCS["botrun_speed_" .. n] = function(e)
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
        minw = 1.05,
        minh = 0.6,
        label = { n .. "X" },
        scale = 0.35,
        col = true,
    })
end

local botrun_orig_main_menu = create_UIBox_main_menu_buttons
function create_UIBox_main_menu_buttons()
    local t = botrun_orig_main_menu()
    -- t.nodes[1] is the C holding the button bar (UI_definitions.lua:6212-6222).
    -- The bar's PLAY/(OPTIONS|QUIT)/COLLECTION buttons are C nodes, which the
    -- UI engine lays out side by side (ui.lua:187-198) — inserting another C
    -- widens the bar past the screen. R children stack vertically instead, so
    -- BOT RUN goes in as its own row above the bar, leaving the bar untouched.
    local holder = t.nodes and t.nodes[1]
    if holder and holder.nodes then
        table.insert(holder.nodes, 1, {
            n = G.UIT.R,
            config = { align = "cm", padding = 0.15, r = 0.1, emboss = 0.1, colour = G.C.L_BLACK },
            nodes = {
                { n = G.UIT.R, config = { align = "cm" }, nodes = {
                    UIBox_button({
                        id = "botrun_button",
                        button = "botrun_click",
                        colour = G.C.GREEN,
                        minw = 3.65,
                        minh = 0.9,
                        label = { "BOT RUN" },
                        scale = 0.45,
                        col = true,
                    }),
                } },
                { n = G.UIT.R, config = { align = "cm", minh = 0.1 }, nodes = {} },
                { n = G.UIT.R, config = { align = "cm" }, nodes = {
                    botrun_speed_button(4, G.C.ORANGE),
                    { n = G.UIT.C, config = { minw = 0.15 }, nodes = {} },
                    botrun_speed_button(8, G.C.RED),
                    { n = G.UIT.C, config = { minw = 0.15 }, nodes = {} },
                    botrun_speed_button(16, G.C.PURPLE),
                } },
            },
        })
        table.insert(holder.nodes, 2, { n = G.UIT.R, config = { minh = 0.15 }, nodes = {} })
    end
    return t
end

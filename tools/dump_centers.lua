-- Dumps the P_CENTERS / P_TAGS prototype tables from the extracted game.lua
-- (reference/lua_source/game.lua) as TSV, for generating + verifying the Rust
-- item registry (sim/core/src/items.rs).
--
-- Run:  tools/oracle/luajit tools/dump_centers.lua \
--         > sim/core/tests/data/centers.tsv
--
-- The script extracts the literal `self.P_CENTERS = { ... }` and
-- `self.P_TAGS = { ... }` table constructors (game.lua:224-258 / :364-743)
-- by brace matching and evaluates them in a stub environment; the tables are
-- pure data (no function calls), so this is exact.

local function read_file(path)
  local f = assert(io.open(path, "r"))
  local s = f:read("*a")
  f:close()
  return s
end

local src = read_file("reference/lua_source/game.lua")

local function extract_table(marker)
  local start = assert(src:find(marker, 1, true), marker)
  local open = assert(src:find("{", start, true))
  local depth, i = 0, open
  while true do
    local c = src:sub(i, i)
    if c == "{" then depth = depth + 1
    elseif c == "}" then
      depth = depth - 1
      if depth == 0 then break end
    elseif c == "'" or c == '"' then
      -- skip string literal
      local q = c
      repeat
        i = i + 1
        c = src:sub(i, i)
      until c == q or i > #src
    elseif c == "-" and src:sub(i + 1, i + 1) == "-" then
      i = src:find("\n", i, true) or #src
    end
    i = i + 1
  end
  local body = src:sub(open, i)
  local chunk = assert(loadstring("return " .. body))
  return setfenv(chunk, {})()
end

local CENTERS = extract_table("self.P_CENTERS = {")
local TAGS = extract_table("self.P_TAGS = {")

local function fmt(v)
  if v == nil then return "" end
  if type(v) == "boolean" then return v and "1" or "0" end
  if type(v) == "number" then
    -- exact decimal for the fractional weights/rates (0.25, 9.6/4, ...)
    return string.format("%.17g", v)
  end
  return tostring(v)
end

-- one row per center: kind-specific columns are left empty when absent
local out = io.stdout
out:write("key\tset\torder\tname\tcost\trarity\tunlocked\tweight\tkind\t" ..
  "extra\tchoose\tmax_highlighted\tmin_highlighted\tmod_conv\tsuit_conv\t" ..
  "hand_type\tsoftlock\tplanets\ttarots\tremove_card\thidden\trequires\tmin_ante\tenhancement_gate\tno_pool_flag\tyes_pool_flag\t" ..
  "blueprint_compat\tperishable_compat\teternal_compat\n")

local keys = {}
for k in pairs(CENTERS) do keys[#keys + 1] = k end
table.sort(keys)
for _, k in ipairs(keys) do
  local v = CENTERS[k]
  if type(v) == "table" and v.set then
    local cfg = v.config or {}
    local extra = cfg.extra
    local extra_s = ""
    if type(extra) == "table" then
      -- only Immolate {destroy=5,dollars=20} and joker tables land here;
      -- serialize the two consumable-relevant fields
      if extra.destroy then extra_s = ("destroy=%d,dollars=%d"):format(extra.destroy, extra.dollars) end
    else
      extra_s = fmt(extra)
    end
    local req = ""
    if type(v.requires) == "table" then req = table.concat(v.requires, ",") end
    out:write(table.concat({
      k, fmt(v.set), fmt(v.order), fmt(v.name), fmt(v.cost), fmt(v.rarity),
      fmt(v.unlocked), fmt(v.weight), fmt(v.kind),
      extra_s, fmt(cfg.choose), fmt(cfg.max_highlighted), fmt(cfg.min_highlighted),
      fmt(cfg.mod_conv), fmt(cfg.suit_conv), fmt(cfg.hand_type), fmt(cfg.softlock),
      fmt(cfg.planets), fmt(cfg.tarots), fmt(cfg.remove_card), fmt(v.hidden), req, "",
      fmt(v.enhancement_gate), fmt(v.no_pool_flag), fmt(v.yes_pool_flag),
      fmt(v.blueprint_compat), fmt(v.perishable_compat), fmt(v.eternal_compat),
    }, "\t") .. "\n")
  end
end

local tkeys = {}
for k in pairs(TAGS) do tkeys[#tkeys + 1] = k end
table.sort(tkeys)
for _, k in ipairs(tkeys) do
  local v = TAGS[k]
  out:write(table.concat({
    k, "Tag", fmt(v.order), fmt(v.name), "", "", "", "", "",
    "", "", "", "", "", "", "", "", "", "", "", "", fmt(v.requires), fmt(v.min_ante), "", "", "",
    "", "", "",
  }, "\t") .. "\n")
end

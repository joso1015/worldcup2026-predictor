"""
Builder for derived data files, using the REAL official 2026 World Cup
group-stage schedule (venues, dates, kickoff times).

Source kickoff times are published in UK time (BST = UTC+1); this script
converts each to the venue's local time using the per-venue UTC offset in
venues.csv, then derives the local date and an expected match-time temperature
from the venue's seasonal climate normals (afternoon high / evening) and whether
the stadium is climate-controlled (indoor). Temperatures are seasonal estimates,
not live forecasts -- matches are weeks out.

Writes:
  - data/groups.csv      group -> team membership
  - data/fixtures.csv    72 group matches with venue, city, country, local date,
                         local kickoff, match day, and expected temperature
  - data/adjustments.csv per-team injury/travel/weather factors (neutral defaults)

Re-run with:  python build_data.py
"""
import csv
import datetime as dt
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

# Normalise schedule team names to our elo_ratings.csv spellings.
NAME = {
    "South Korea": "South Korea", "Czech Republic": "Czechia",
    "Bosnia & Herzegovina": "Bosnia and Herzegovina", "USA": "United States",
    "Turkey": "Turkiye", "Curacao": "Curacao", "Cape Verde": "Cape Verde",
    "DR Congo": "DR Congo", "Ivory Coast": "Ivory Coast",
}

# Schedule city -> stadium name (must match venues.csv 'venue').
CITY_VENUE = {
    "Mexico City": "Estadio Azteca", "Zapopan": "Estadio Akron",
    "Guadalupe": "Estadio BBVA", "Los Angeles": "SoFi Stadium",
    "Santa Clara": "Levi's Stadium", "Seattle": "Lumen Field",
    "Kansas City": "Arrowhead Stadium", "Arlington": "AT&T Stadium",
    "Houston": "NRG Stadium", "Atlanta": "Mercedes-Benz Stadium",
    "Miami": "Hard Rock Stadium", "Foxborough": "Gillette Stadium",
    "Philadelphia": "Lincoln Financial Field", "New Jersey": "MetLife Stadium",
    "Toronto": "BMO Field", "Vancouver": "BC Place",
}

# Official group-stage schedule: (group, home, away, city, BST date, BST HH:MM).
# Kickoff times are UK time (BST, UTC+1) as published by the source.
SCHEDULE = [
    ("A", "Mexico", "South Africa", "Mexico City", "2026-06-11", "20:00"),
    ("A", "South Korea", "Czech Republic", "Zapopan", "2026-06-12", "03:00"),
    ("B", "Canada", "Bosnia & Herzegovina", "Toronto", "2026-06-12", "20:00"),
    ("D", "USA", "Paraguay", "Los Angeles", "2026-06-13", "02:00"),
    ("B", "Qatar", "Switzerland", "Santa Clara", "2026-06-13", "20:00"),
    ("C", "Brazil", "Morocco", "New Jersey", "2026-06-13", "23:00"),
    ("C", "Haiti", "Scotland", "Foxborough", "2026-06-14", "02:00"),
    ("D", "Australia", "Turkey", "Vancouver", "2026-06-14", "05:00"),
    ("E", "Germany", "Curacao", "Houston", "2026-06-14", "18:00"),
    ("F", "Netherlands", "Japan", "Arlington", "2026-06-14", "21:00"),
    ("E", "Ivory Coast", "Ecuador", "Philadelphia", "2026-06-15", "00:00"),
    ("F", "Sweden", "Tunisia", "Guadalupe", "2026-06-15", "03:00"),
    ("H", "Spain", "Cape Verde", "Atlanta", "2026-06-15", "17:00"),
    ("G", "Belgium", "Egypt", "Seattle", "2026-06-15", "20:00"),
    ("H", "Saudi Arabia", "Uruguay", "Miami", "2026-06-15", "23:00"),
    ("G", "Iran", "New Zealand", "Los Angeles", "2026-06-16", "02:00"),
    ("I", "France", "Senegal", "New Jersey", "2026-06-16", "20:00"),
    ("I", "Iraq", "Norway", "Foxborough", "2026-06-16", "23:00"),
    ("J", "Argentina", "Algeria", "Kansas City", "2026-06-17", "02:00"),
    ("J", "Austria", "Jordan", "Santa Clara", "2026-06-17", "05:00"),
    ("K", "Portugal", "DR Congo", "Houston", "2026-06-17", "18:00"),
    ("L", "England", "Croatia", "Arlington", "2026-06-17", "21:00"),
    ("L", "Ghana", "Panama", "Toronto", "2026-06-18", "00:00"),
    ("K", "Uzbekistan", "Colombia", "Mexico City", "2026-06-18", "03:00"),
    ("A", "Czech Republic", "South Africa", "Atlanta", "2026-06-18", "17:00"),
    ("B", "Switzerland", "Bosnia & Herzegovina", "Los Angeles", "2026-06-18", "20:00"),
    ("B", "Canada", "Qatar", "Vancouver", "2026-06-18", "23:00"),
    ("A", "Mexico", "South Korea", "Zapopan", "2026-06-19", "02:00"),
    ("D", "USA", "Australia", "Seattle", "2026-06-19", "20:00"),
    ("C", "Scotland", "Morocco", "Foxborough", "2026-06-19", "23:00"),
    ("C", "Brazil", "Haiti", "Philadelphia", "2026-06-20", "01:30"),
    ("D", "Turkey", "Paraguay", "Santa Clara", "2026-06-20", "04:00"),
    ("F", "Netherlands", "Sweden", "Houston", "2026-06-20", "18:00"),
    ("E", "Germany", "Ivory Coast", "Toronto", "2026-06-20", "21:00"),
    ("E", "Ecuador", "Curacao", "Kansas City", "2026-06-21", "01:00"),
    ("F", "Tunisia", "Japan", "Guadalupe", "2026-06-21", "05:00"),
    ("H", "Spain", "Saudi Arabia", "Atlanta", "2026-06-21", "17:00"),
    ("G", "Belgium", "Iran", "Los Angeles", "2026-06-21", "20:00"),
    ("H", "Uruguay", "Cape Verde", "Miami", "2026-06-21", "23:00"),
    ("G", "New Zealand", "Egypt", "Vancouver", "2026-06-22", "02:00"),
    ("J", "Argentina", "Austria", "Arlington", "2026-06-22", "18:00"),
    ("I", "France", "Iraq", "Philadelphia", "2026-06-22", "22:00"),
    ("I", "Norway", "Senegal", "Toronto", "2026-06-23", "01:00"),
    ("J", "Jordan", "Algeria", "Santa Clara", "2026-06-23", "04:00"),
    ("K", "Portugal", "Uzbekistan", "Houston", "2026-06-23", "18:00"),
    ("L", "England", "Ghana", "Foxborough", "2026-06-23", "21:00"),
    ("L", "Panama", "Croatia", "Foxborough", "2026-06-24", "00:00"),
    ("K", "Colombia", "DR Congo", "Zapopan", "2026-06-24", "03:00"),
    ("B", "Switzerland", "Canada", "Vancouver", "2026-06-24", "20:00"),
    ("B", "Bosnia & Herzegovina", "Qatar", "Seattle", "2026-06-24", "20:00"),
    ("C", "Morocco", "Haiti", "Atlanta", "2026-06-24", "23:00"),
    ("C", "Scotland", "Brazil", "Miami", "2026-06-24", "23:00"),
    ("A", "South Africa", "South Korea", "Guadalupe", "2026-06-25", "02:00"),
    ("A", "Czech Republic", "Mexico", "Mexico City", "2026-06-25", "02:00"),
    ("E", "Curacao", "Ivory Coast", "Philadelphia", "2026-06-25", "21:00"),
    ("E", "Ecuador", "Germany", "New Jersey", "2026-06-25", "21:00"),
    ("F", "Tunisia", "Netherlands", "Kansas City", "2026-06-26", "00:00"),
    ("F", "Japan", "Sweden", "Arlington", "2026-06-26", "00:00"),
    ("D", "Turkey", "USA", "Los Angeles", "2026-06-26", "03:00"),
    ("D", "Paraguay", "Australia", "Santa Clara", "2026-06-26", "03:00"),
    ("I", "Norway", "France", "Foxborough", "2026-06-26", "20:00"),
    ("I", "Senegal", "Iraq", "Toronto", "2026-06-26", "20:00"),
    ("H", "Cape Verde", "Saudi Arabia", "Houston", "2026-06-27", "01:00"),
    ("H", "Uruguay", "Spain", "Zapopan", "2026-06-27", "01:00"),
    ("G", "New Zealand", "Belgium", "Vancouver", "2026-06-27", "04:00"),
    ("G", "Egypt", "Iran", "Seattle", "2026-06-27", "04:00"),
    ("L", "Panama", "England", "New Jersey", "2026-06-27", "22:00"),
    ("L", "Croatia", "Ghana", "Philadelphia", "2026-06-27", "22:00"),
    ("K", "Colombia", "Portugal", "Miami", "2026-06-28", "00:30"),
    ("K", "DR Congo", "Uzbekistan", "Atlanta", "2026-06-28", "00:30"),
    ("J", "Algeria", "Austria", "Kansas City", "2026-06-28", "03:00"),
    ("J", "Jordan", "Argentina", "Arlington", "2026-06-28", "03:00"),
]


def load_venue_meta():
    meta = {}
    with open(os.path.join(DATA, "venues.csv"), newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            meta[r["venue"]] = r
    return meta


def expected_temp(meta, local_hour):
    """Seasonal estimate of match-time temperature (deg C)."""
    if int(meta["indoor"]):
        return 22  # climate-controlled dome
    high = float(meta["afternoon_high_c"])
    eve = float(meta["evening_c"])
    h = local_hour
    if 12 <= h <= 16:
        return round(high)
    if 17 <= h <= 20:
        return round((high + eve) / 2)
    if 21 <= h <= 23 or 0 <= h <= 6:
        return round(eve - 3)
    return round((high + eve) / 2 - 2)  # morning


def haversine_km(lat1, lon1, lat2, lon2):
    import math
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return round(2 * r * math.asin(math.sqrt(a)), 0)


def travel_factor(rest_h, dist_km):
    """Transparent rest/travel multiplier on expected goals. Short rest (<96h)
    and long travel each shave a little; neutral when there's no prior match."""
    f = 1.0
    if rest_h is not None and rest_h < 96:
        f -= min((96 - rest_h) / 96.0, 1.0) * 0.08      # up to -8% for no rest
    if dist_km:
        f -= min(dist_km / 8000.0, 1.0) * 0.05          # up to -5% for long haul
    return round(max(f, 0.85), 3)


def main():
    venue_meta = load_venue_meta()

    # group membership from elo_ratings.csv
    groups = {}
    host_of = {}
    with open(os.path.join(DATA, "elo_ratings.csv"), newline="", encoding="utf-8") as f:
        teams = list(csv.DictReader(f))
    for row in teams:
        groups.setdefault(row["group"], []).append(row["team"])
        host_of[row["team"]] = row["host_country"].strip()

    with open(os.path.join(DATA, "groups.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["group", "team"])
        for g in sorted(groups):
            for t in groups[g]:
                w.writerow([g, t])

    # match-day counter per group (1/2/3 across the three rounds)
    md_count = {}
    rows = []
    for n, (g, home, away, city, bst_date, bst_hhmm) in enumerate(SCHEDULE, start=1):
        home = NAME.get(home, home)
        away = NAME.get(away, away)
        venue = CITY_VENUE[city]
        meta = venue_meta[venue]
        offset = int(meta["utc_offset"])

        # BST (UTC+1) -> local: local = BST - 1h + offset
        bst = dt.datetime.strptime(f"{bst_date} {bst_hhmm}", "%Y-%m-%d %H:%M")
        local = bst + dt.timedelta(hours=(offset - 1))
        temp = expected_temp(meta, local.hour)

        # determine which of the group's 3 match days this is
        key = g
        md_count[key] = md_count.get(key, 0)
        seen_pairs = md_count[key]
        match_day = seen_pairs // 2 + 1
        md_count[key] += 1

        rows.append(dict(
            match_no=n, group=g, match_day=match_day,
            date_local=local.strftime("%Y-%m-%d"),
            kickoff_local=local.strftime("%H:%M"),
            home_team=home, away_team=away,
            city=meta["city"], venue=venue, country=meta["country"],
            indoor=meta["indoor"], expected_temp_c=temp,
            _utc=bst - dt.timedelta(hours=1),  # absolute time for rest calc
            _lat=float(meta["lat"]), _lon=float(meta["lon"]),
        ))

    # Second pass: each team's rest hours and travel distance from its previous
    # match, plus the derived travel/rest factor (fed into the model's odds).
    appearances = {}
    for r in rows:
        for side in ("home_team", "away_team"):
            appearances.setdefault(r[side], []).append(r)
    for r in rows:
        for side, pfx in (("home_team", "home"), ("away_team", "away")):
            team = r[side]
            seq = sorted(appearances[team], key=lambda x: x["_utc"])
            k = seq.index(r)
            if k == 0:
                rest_h, dist = "", ""
                fac = 1.0
            else:
                prev = seq[k - 1]
                rest_h = round((r["_utc"] - prev["_utc"]).total_seconds() / 3600.0)
                dist = haversine_km(prev["_lat"], prev["_lon"], r["_lat"], r["_lon"])
                fac = travel_factor(rest_h, dist)
            r[f"{pfx}_rest_h"] = rest_h
            r[f"{pfx}_dist_km"] = dist
            r[f"{pfx}_travel_factor"] = fac

    cols = ["match_no", "group", "match_day", "date_local", "kickoff_local",
            "home_team", "away_team", "city", "venue", "country", "indoor",
            "expected_temp_c", "home_rest_h", "home_dist_km", "home_travel_factor",
            "away_rest_h", "away_dist_km", "away_travel_factor"]
    with open(os.path.join(DATA, "fixtures.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    with open(os.path.join(DATA, "adjustments.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["team", "injury_factor", "travel_rest_factor",
                    "weather_factor", "notes"])
        for row in teams:
            w.writerow([row["team"], 1.00, 1.00, 1.00, ""])

    print(f"Wrote groups.csv, fixtures.csv ({len(rows)} matches), adjustments.csv")


if __name__ == "__main__":
    main()

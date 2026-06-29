"""
Fills data/squads.csv with the official 2026 World Cup final squads.

Two modes:
  * Online (default): downloads the full squad list from Wikipedia's structured
    wikitext (action=raw, not truncated) and parses all 48 teams via the
    {{nat fs player|...}} templates. Run on any machine with internet:
        python build_squads.py
  * Offline fallback: if the download is blocked, it writes the squads embedded
    below (Groups A & B, verified) so the feature still works.

Real rosters only — no invented players. Source: FIFA squad lists as published
on Wikipedia (June 2026).
"""
import csv
import os
import re
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
RAW_URL = "https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_squads?action=raw"

G, D, M, F = "Goalkeeper", "Defender", "Midfielder", "Forward"
POS = {"GK": G, "DF": D, "MF": M, "FW": F}

# Map Wikipedia's team headings to the names used in elo_ratings.csv.
TEAM_ALIAS = {
    "Czech Republic": "Czechia", "Türkiye": "Turkiye", "Turkey": "Turkiye",
    "Curaçao": "Curacao", "Ivory Coast": "Ivory Coast",
    "Côte d'Ivoire": "Ivory Coast", "United States": "United States",
    "South Korea": "South Korea", "DR Congo": "DR Congo",
    "Cape Verde": "Cape Verde",
}

VALID_TEAMS = None  # loaded from elo_ratings.csv at runtime


def _link_text(s):
    """[[Link|Display]] / [[Link]] / plain -> readable text."""
    s = s.strip()
    m = re.search(r"\[\[([^\]]+)\]\]", s)
    if m:
        inner = m.group(1)
        return inner.split("|")[-1].strip()
    return re.sub(r"\{\{[^}]*\}\}", "", s).strip()


def download_squads():
    req = urllib.request.Request(RAW_URL, headers={"User-Agent": "wc2026-predictor"})
    with urllib.request.urlopen(req, timeout=60) as r:
        wikitext = r.read().decode("utf-8", "ignore")

    squads = {}
    current = None
    # team headings look like ===Mexico=== (level-3); group headings ==Group A==
    for line in wikitext.splitlines():
        hm = re.match(r"\s*===\s*([^=].*?)\s*===\s*$", line)
        if hm:
            name = _link_text(hm.group(1))
            name = TEAM_ALIAS.get(name, name)
            current = name if name in VALID_TEAMS else None
            if current:
                squads.setdefault(current, [])
            continue
        if current and "nat fs player" in line:
            params = dict(re.findall(r"\|\s*([a-zA-Z0-9_]+)\s*=\s*([^|}]+)", line))
            pos = POS.get((params.get("pos", "") or "").strip().upper())
            name = _link_text(params.get("name", ""))
            club = _link_text(params.get("club", ""))
            if pos and name:
                squads[current].append((pos, name, club))
    return {k: v for k, v in squads.items() if v}


# ---- Offline fallback: verified Groups A & B (8 teams) ----------------------
EMBEDDED = {
"Czechia": [
 (G,"Matěj Kovář","PSV Eindhoven"),(D,"David Zima","Slavia Prague"),
 (D,"Tomáš Holeš","Slavia Prague"),(D,"Robin Hranáč","TSG Hoffenheim"),
 (D,"Vladimír Coufal","TSG Hoffenheim"),(D,"Štěpán Chaloupek","Slavia Prague"),
 (D,"Ladislav Krejčí","Wolverhampton Wanderers"),(M,"Vladimír Darida","Hradec Králové"),
 (F,"Adam Hložek","TSG Hoffenheim"),(F,"Patrik Schick","Bayer Leverkusen"),
 (F,"Jan Kuchta","Sparta Prague"),(M,"Lukáš Červ","Viktoria Plzeň"),
 (F,"Mojmír Chytil","Slavia Prague"),(D,"David Jurásek","Slavia Prague"),
 (F,"Pavel Šulc","Lyon"),(G,"Jindřich Staněk","Slavia Prague"),
 (M,"Lukáš Provod","Slavia Prague"),(M,"Michal Sadílek","Slavia Prague"),
 (F,"Tomáš Chorý","Slavia Prague"),(D,"Jaroslav Zelený","Sparta Prague"),
 (D,"David Douděra","Slavia Prague"),(M,"Tomáš Souček","West Ham United"),
 (G,"Lukáš Horníček","Braga"),(M,"Alexandr Sojka","Viktoria Plzeň"),
 (M,"Hugo Sochůrek","Sparta Prague"),(F,"Denis Višinský","Viktoria Plzeň"),
],
"Mexico": [
 (G,"Raúl Rangel","Guadalajara"),(D,"Jorge Sánchez","PAOK"),
 (D,"César Montes","Lokomotiv Moscow"),(D,"Edson Álvarez","Fenerbahçe"),
 (D,"Johan Vásquez","Genoa"),(M,"Érik Lira","Cruz Azul"),
 (M,"Luis Romo","Guadalajara"),(M,"Álvaro Fidalgo","Real Betis"),
 (F,"Raúl Jiménez","Fulham"),(F,"Alexis Vega","Toluca"),
 (F,"Santiago Giménez","Milan"),(G,"Carlos Acevedo","Santos Laguna"),
 (G,"Guillermo Ochoa","AEL Limassol"),(F,"Armando González","Guadalajara"),
 (D,"Israel Reyes","América"),(F,"Julián Quiñones","Al-Qadsiah"),
 (M,"Orbelín Pineda","AEK Athens"),(M,"Obed Vargas","Atlético Madrid"),
 (M,"Gilberto Mora","Tijuana"),(D,"Mateo Chávez","AZ"),
 (F,"César Huerta","Anderlecht"),(F,"Guillermo Martínez","Pumas"),
 (D,"Jesús Gallardo","Toluca"),(M,"Luis Chávez","Dynamo Moscow"),
 (F,"Roberto Alvarado","Guadalajara"),(M,"Brian Gutiérrez","Guadalajara"),
],
"South Africa": [
 (G,"Ronwen Williams","Mamelodi Sundowns"),(D,"Thabang Matuludi","Polokwane City"),
 (D,"Khulumani Ndamane","Mamelodi Sundowns"),(M,"Teboho Mokoena","Mamelodi Sundowns"),
 (M,"Thalente Mbatha","Orlando Pirates"),(D,"Aubrey Modiba","Mamelodi Sundowns"),
 (F,"Oswin Appollis","Orlando Pirates"),(F,"Tshepang Moremi","Orlando Pirates"),
 (F,"Lyle Foster","Burnley"),(F,"Relebohile Mofokeng","Orlando Pirates"),
 (M,"Themba Zwane","Mamelodi Sundowns"),(F,"Thapelo Maseko","AEL Limassol"),
 (M,"Sphephelo Sithole","Tondela"),(D,"Mbekezeli Mbokazi","Chicago Fire FC"),
 (F,"Iqraam Rayners","Mamelodi Sundowns"),(G,"Sipho Chaine","Orlando Pirates"),
 (F,"Evidence Makgopa","Orlando Pirates"),(D,"Samukele Kabini","Molde"),
 (D,"Nkosinathi Sibisi","Orlando Pirates"),(D,"Khuliso Mudau","Mamelodi Sundowns"),
 (D,"Ime Okon","Hannover 96"),(G,"Ricardo Goss","Siwelele"),
 (M,"Jayden Adams","Mamelodi Sundowns"),(D,"Olwethu Makhanya","Philadelphia Union"),
 (F,"Kamogelo Sebelebele","Orlando Pirates"),(D,"Bradley Cross","Kaizer Chiefs"),
],
"South Korea": [
 (G,"Kim Seung-gyu","FC Tokyo"),(D,"Lee Han-beom","Midtjylland"),
 (M,"Lee Gi-hyuk","Gangwon FC"),(D,"Kim Min-jae","Bayern Munich"),
 (D,"Kim Tae-hyeon","Kashima Antlers"),(M,"Hwang In-beom","Feyenoord"),
 (F,"Son Heung-min","Los Angeles FC"),(M,"Paik Seung-ho","Birmingham City"),
 (F,"Cho Gue-sung","Midtjylland"),(M,"Lee Jae-sung","Mainz 05"),
 (M,"Hwang Hee-chan","Wolverhampton Wanderers"),(G,"Song Bum-keun","Jeonbuk Hyundai Motors"),
 (D,"Lee Tae-seok","Austria Wien"),(D,"Cho Wi-je","Jeonbuk Hyundai Motors"),
 (D,"Kim Moon-hwan","Daejeon Hana Citizen"),(D,"Park Jin-seob","Zhejiang"),
 (M,"Bae Jun-ho","Stoke City"),(F,"Oh Hyeon-gyu","Beşiktaş"),
 (M,"Lee Kang-in","Paris Saint-Germain"),(M,"Yang Hyun-jun","Celtic"),
 (G,"Jo Hyeon-woo","Ulsan HD"),(D,"Seol Young-woo","Red Star Belgrade"),
 (D,"Jens Castrop","Borussia Mönchengladbach"),(M,"Kim Jin-gyu","Jeonbuk Hyundai Motors"),
 (M,"Eom Ji-sung","Swansea City"),(M,"Lee Dong-gyeong","Ulsan HD"),
],
"Bosnia and Herzegovina": [
 (G,"Nikola Vasilj","FC St. Pauli"),(D,"Nihad Mujakić","Gaziantep"),
 (D,"Dennis Hadžikadunić","Sampdoria"),(D,"Tarik Muharemović","Sassuolo"),
 (D,"Sead Kolašinac","Atalanta"),(M,"Benjamin Tahirović","Brøndby"),
 (D,"Amar Dedić","Benfica"),(M,"Armin Gigović","Young Boys"),
 (F,"Samed Baždar","Jagiellonia Białystok"),(F,"Ermedin Demirović","VfB Stuttgart"),
 (F,"Edin Džeko","Schalke 04"),(G,"Mladen Jurkas","Borac Banja Luka"),
 (M,"Ivan Bašić","Astana"),(M,"Ivan Šunjić","Pafos"),
 (M,"Amar Memić","Viktoria Plzeň"),(M,"Amir Hadžiahmetović","Hull City"),
 (M,"Dženis Burnić","Karlsruher SC"),(D,"Nikola Katić","Schalke 04"),
 (F,"Kerim Alajbegović","Red Bull Salzburg"),(F,"Esmir Bajraktarević","PSV Eindhoven"),
 (D,"Stjepan Radeljić","Rijeka"),(G,"Martin Zlomislić","Rijeka"),
 (F,"Haris Tabaković","Borussia Mönchengladbach"),(D,"Nidal Čelik","Lens"),
 (F,"Jovo Lukić","Universitatea Cluj"),(M,"Ermin Mahmić","Slovan Liberec"),
],
"Canada": [
 (G,"Dayne St. Clair","Inter Miami CF"),(D,"Alistair Johnston","Celtic"),
 (D,"Alfie Jones","Middlesbrough"),(D,"Luc de Fougerolles","Dender"),
 (D,"Joel Waterman","Chicago Fire FC"),(M,"Mathieu Choinière","Los Angeles FC"),
 (M,"Stephen Eustáquio","Los Angeles FC"),(M,"Ismaël Koné","Sassuolo"),
 (F,"Cyle Larin","Southampton"),(F,"Jonathan David","Juventus"),
 (M,"Liam Millar","Hull City"),(F,"Tani Oluwaseyi","Villarreal"),
 (D,"Derek Cornelius","Rangers"),(M,"Jacob Shaffelburg","Los Angeles FC"),
 (D,"Moïse Bombito","Nice"),(G,"Maxime Crépeau","Orlando City SC"),
 (F,"Tajon Buchanan","Villarreal"),(G,"Owen Goodman","Barnsley"),
 (D,"Alphonso Davies","Bayern Munich"),(F,"Ali Ahmed","Norwich City"),
 (M,"Jonathan Osorio","Toronto FC"),(D,"Richie Laryea","Toronto FC"),
 (D,"Niko Sigur","Hajduk Split"),(F,"Promise David","Union Saint-Gilloise"),
 (M,"Nathan Saliba","Anderlecht"),
],
"Qatar": [
 (G,"Mahmud Abunada","Al-Rayyan"),(D,"Pedro Miguel","Al-Sadd"),
 (D,"Lucas Mendes","Al-Wakrah"),(D,"Issa Laye","Al-Arabi"),
 (D,"Jassem Gaber","Al-Rayyan"),(M,"Abdulaziz Hatem","Al-Rayyan"),
 (F,"Ahmed Alaaeldin","Al-Rayyan"),(F,"Edmilson Junior","Al-Duhail"),
 (F,"Mohammed Muntari","Al-Gharafa"),(F,"Hassan Al-Haydos","Al-Sadd"),
 (F,"Akram Afif","Al-Sadd"),(M,"Karim Boudiaf","Al-Duhail"),
 (D,"Ayoub Al-Oui","Al-Gharafa"),(D,"Homam Ahmed","Cultural Leonesa"),
 (F,"Yusuf Abdurisag","Al-Wakrah"),(D,"Boualem Khoukhi","Al-Sadd"),
 (M,"Ahmed Al-Ganehi","Al-Gharafa"),(D,"Sultan Al-Brake","Al-Duhail"),
 (F,"Almoez Ali","Al-Duhail"),(M,"Ahmed Fathy","Al-Arabi"),
 (G,"Salah Zakaria","Al-Duhail"),(G,"Meshaal Barsham","Al-Sadd"),
 (M,"Assim Madibo","Al-Wakrah"),(F,"Tahsin Jamshid","Al-Duhail"),
 (D,"Al-Hashmi Al-Hussain","Al-Arabi"),(F,"Mohamed Manai","Al-Shamal"),
],
}


def main():
    global VALID_TEAMS
    with open(os.path.join(DATA, "elo_ratings.csv"), newline="", encoding="utf-8") as f:
        VALID_TEAMS = {r["team"] for r in csv.DictReader(f)}

    squads, source = EMBEDDED, "embedded (Groups A & B)"
    try:
        dl = download_squads()
        if len(dl) >= 12:                 # got a real, broad pull
            merged = dict(EMBEDDED)
            merged.update(dl)
            squads, source = merged, f"Wikipedia ({len(dl)} teams downloaded)"
    except Exception as e:
        print(f"  (download unavailable: {type(e).__name__}; using embedded squads)")

    path = os.path.join(DATA, "squads.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["team", "player", "position", "club"])
        for team in sorted(squads):
            for pos, name, club in squads[team]:
                w.writerow([team, name, pos, club])
    n = sum(len(v) for v in squads.values())
    print(f"Wrote squads.csv — {len(squads)} teams, {n} players · source: {source}")


if __name__ == "__main__":
    main()
